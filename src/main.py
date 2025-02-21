import sys
import os
import operator
import tiktoken
from typing import Annotated, List, Tuple, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatMaritalk
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pathlib import Path

from utils import RichLogger, display_graph

load_dotenv("../credentials.env")
logger = RichLogger("pipeline")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# tools...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@tool
def tokens_from_string(string: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# utility functions...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_file_contents(file_paths: List[str], vector_store: Chroma) -> dict:
    file_contents = {}
    all_splits = []
    
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        try:
            file_extension = Path(path).suffix.lower()
            documents = []
            
            if file_extension == '.pdf':
                logger.info(f"Loading PDF file: {path}")
                loader = PyPDFLoader(path)
                documents = loader.load()
            elif file_extension in ['.docx', '.doc']:
                logger.info(f"Loading Word document: {path}")
                loader = Docx2txtLoader(path)
                documents = loader.load()
            else:
                logger.info(f"Loading text file: {path}")
                loader = TextLoader(path)
                documents = loader.load()
            
            file_contents[path] = "\n".join(doc.page_content for doc in documents)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 4098,
                chunk_overlap = 256,
                length_function = len,
                is_separator_regex = False,
            )
            splits = text_splitter.split_documents(documents)
            
            # Add page numbers and source to metadata
            for i, split_doc in enumerate(splits):
                original_page = split_doc.metadata.get('page', None)
                if original_page is not None:
                    split_doc.metadata['page_number'] = original_page
                else:
                    split_doc.metadata['page_number'] = i
                split_doc.metadata['source'] = path
            
            all_splits.extend(splits)
            
            # Calculate tokens per chunk
            tokens_per_chunk = [tokens_from_string(doc.page_content) for doc in splits]
            logger.info(f"Tokens per chunk: {tokens_per_chunk}")
            
            logger.success(f"Successfully processed: {path}")
            
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
            continue
    
    # Add all documents to vector store at once
    if all_splits:
        vector_store.add_documents(all_splits)
    
    logger.section("File Loading Summary")
    logger.dict({
        "total_files": len(file_paths),
        "processed_files": len(file_contents),
        "processed_paths": list(file_contents.keys())
    })
    return file_contents


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# initializing the vector store...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PERSIST_DIRECTORY = "/home/dusoudeth/Documentos/github/baxi-renmin-wansui/db/vector_store"
# remove existing vector store
if os.path.exists(PERSIST_DIRECTORY):
    os.system(f"rm -rf {PERSIST_DIRECTORY}")
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# initializing the language model and agents...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
llm_gpt_4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_sabia_3 = ChatMaritalk(
    model="sabia-3",
    api_key=os.getenv("MARITACA_KEY"),
    temperature=0.01
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{messages}")
])

# Create agents
agent_summarizer = create_react_agent(
    model=llm_sabia_3,
    tools=list(),
    state_modifier=prompt
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the pipeline state...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class PipelineState(TypedDict):
    messages: List[HumanMessage]
    next_chapters: List[Tuple[int, str]]
    past_chapters: Annotated[List[Tuple[int, str]], operator.add]
    summary: Annotated[List[str], operator.add]
    active_agent: str
    processing_complete: bool

class IsTheSame(BaseModel):
    is_the_same: bool = Field(description="True if the two strings are very similar, False otherwise")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# setting up the planning...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
EVALUATION_PROMPT = """
QUESTION: 

Is the new summary similar or contained in the previous summary?

SUMMARY SO FAR:

{old}

NEW SUMMARY:
{new}
"""


def plan_step(state: PipelineState):
    """Create development plan"""
    if not state.get("processing_complete"):
        if (len(state["next_chapters"]) == 0) and (len(state["past_chapters"]) == 0):
            # getting documents from vector store...
            result = vector_store.get()
            documents = list()
            if isinstance(result, dict):
                for i, (doc, metadata) in enumerate(zip(result.get('documents', []), result.get('metadatas', []))):
                    page_number = metadata.get('page_number', i) if metadata else i
                    documents.append({
                        'page_number': page_number,
                        'content': doc
                    })
            # sorting documents by page number...
            sorted_documents = sorted(documents, key=lambda x: x['page_number'])
            # creating initial chapters list
            initial_chapters = []
            for doc in sorted_documents:
                preview = doc['content'][:256] if doc['content'] else ''
                initial_chapters.append((doc['page_number'], preview))
            return {
                "next_chapters": initial_chapters,
                "active_agent": "summarizer",
                "processing_complete": False
            }
    return {"active_agent": "summarizer"}


def should_end(state: PipelineState):
    logger.info(f"Next chapters: {state['next_chapters']}")
    logger.info(f"Past chapters: {state['past_chapters']}")
    if (len(state["next_chapters"]) == 0 and len(state["past_chapters"]) > 0):
        logger.info("Processing complete.\n FINAL SUMMARY: \n" + "\n".join(state["summary"]))
        # overwrite the final summary into a file
        with open("summary.txt", "w") as f:
            f.write("\n".join(state["summary"]))
        return END
    else:
        return "summarizer"


def summarize_step(state: PipelineState):
    """Summarize the current chapter"""
    if len(state["next_chapters"]) == 0:
        return {
            "active_agent": "planner",
            "processing_complete": True
        }
    current_chapter = state["next_chapters"][0]
    current_chapter_str = current_chapter[1]
    # rag
    relevant_docs = vector_store.similarity_search(current_chapter_str, k=1)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    # task
    task_str = f"""
    {state["messages"][-1].content}, in Portuguese:

    {context}

    Please, your output needs to be only the summary content, don't say anything else.
    """
    # agent :: summarizer
    agent_response = agent_summarizer.invoke({
        "messages": [("user", task_str)]
    })
    remaining_chapters = state["next_chapters"][1:]
    past_chapter = [current_chapter]
    new_summary = [agent_response["messages"][-1].content]
    # agent :: evaluator
    evaluation_chain = ChatPromptTemplate.from_template(EVALUATION_PROMPT) | llm_gpt_4o_mini.with_structured_output(IsTheSame)
    evaluation_response = evaluation_chain.invoke({
        "old" : ' '.join(state["summary"]),
        "new" : new_summary[0]
    })
    if evaluation_response.is_the_same:
        logger.warning("The new summary is very similar to the previous summary.")
        return {
            "next_chapters": remaining_chapters,
            "past_chapters": past_chapter,
            "active_agent": "planner",
            "processing_complete": False
        }
    logger.info(f"TASK :: {task_str}")
    logger.info(f"RESPONSE :: {agent_response['messages'][-1].content}")
    return {
        "next_chapters": remaining_chapters,
        "past_chapters": past_chapter,
        "summary": new_summary,
        "active_agent": "planner",
        "processing_complete": False
    }



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the workflow...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
workflow = StateGraph(PipelineState)
# adding nodes... 
workflow.add_node("planner", plan_step)
workflow.add_node("summarizer", summarize_step)
# adding edges...
workflow.add_edge(START, "planner")
workflow.add_edge("summarizer", "planner")
# adding conditional edges...
workflow.add_conditional_edges("planner", should_end, ["summarizer", END])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# compiling the workflow and displaying the graph...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
app = workflow.compile()

display_graph(app)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# setting up the pipeline...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def run_pipeline(query: str, file_paths: List[str]):
    """Run the development pipeline"""
    print(f"Loading file contents from: {file_paths}")
    _ = load_file_contents(file_paths, vector_store)
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_chapters": list(),
        "past_chapters": list(),
        "summary": list(),
        "active_agent": "planner",
        "processing_complete": False
    }    
    print("running pipeline...")
    try:
        app.invoke(initial_state, config={"recursion_limit": 256})
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")


if __name__ == "__main__":
    query = """
        Please summarize the document content.
    """
    file_paths = [
        # "/home/dusoudeth/Calibre Library/Joao Ubaldo Ribeiro/Viva o povo brasileiro (249)/Viva o povo brasileiro - Joao Ubaldo Ribeiro.pdf"
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_acordo .docx",
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_DO_PLANO_DE_CONVIÊNCIA_IDEAL.docx",
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_plano_de_convivencia .docx",
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_plano_de_convivencia_modelo.docx",
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_plano_de_parentalidade.docx",
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_Proposta_de_acordo.docx",
        "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_tese_doutorado.docx",
        # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/plano_de_parentalidade_da_isa.docx",
    ]
    run_pipeline(query, file_paths)