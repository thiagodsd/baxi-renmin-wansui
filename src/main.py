import sys
import os
import asyncio
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

@tool
def tokens_from_string(string: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

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
                chunk_size=4098,
                chunk_overlap=25,
                length_function=len,
                is_separator_regex=False,
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

class PipelineState(TypedDict):
    messages: List[HumanMessage]
    next_chapters: List[Tuple[int, str]]
    past_chapters: Annotated[List[Tuple[int, str]], operator.add]
    summary: Annotated[List[str], operator.add]
    active_agent: str
    processing_complete: bool


class IsTheSame(BaseModel):
    is_the_same: bool = Field(description="True if the two strings are very similar, False otherwise")


async def plan_step(state: PipelineState):
    """Create development plan"""
    if not state.get("processing_complete"):
        if (len(state["next_chapters"]) == 0) and (len(state["past_chapters"]) == 0):
            # Get documents from vector store
            result = vector_store.get()
            
            documents = []
            if isinstance(result, dict):
                for i, (doc, metadata) in enumerate(zip(result.get('documents', []), result.get('metadatas', []))):
                    page_number = metadata.get('page_number', i) if metadata else i
                    documents.append({
                        'page_number': page_number,
                        'content': doc
                    })
            
            # Sort documents by page number
            sorted_documents = sorted(documents, key=lambda x: x['page_number'])
            
            # Create initial chapters list
            initial_chapters = []
            for doc in sorted_documents:
                preview = doc['content'][:256] if doc['content'] else ''
                initial_chapters.append((doc['page_number'], preview))
            
            return {
                "next_chapters": initial_chapters,
                "active_agent": "summarizer",
                "processing_complete": False
            }
    
    # Preserve existing processing_complete state
    return {"active_agent": "summarizer"}

def should_end(state: PipelineState):
    if state.get("processing_complete", False) or (len(state["next_chapters"]) == 0 and len(state["past_chapters"]) > 0):
        return END
    else:
        return "summarizer"

async def summarize_step(state: PipelineState):
    """Summarize the current chapter"""
    if len(state["next_chapters"]) == 0:
        return {
            "active_agent": "planner",
            "processing_complete": True
        }
    
    current_chapter = state["next_chapters"][0]
    current_chapter_str = current_chapter[1]
    
    # Perform RAG query
    relevant_docs = vector_store.similarity_search(current_chapter_str, k=3)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    
    # Create the task string
    task_str = f"""
    Please summarize the excerpts of text from the book "Viva o povo brasileiro" by João Ubaldo Ribeiro, in Portuguese:

    {context}

    Please, your output needs to be only the summary content, don't say anything else.
    """
    
    # Get agent response
    agent_response = await agent_summarizer.ainvoke({
        "messages": [("user", task_str)]
    })
    
    remaining_chapters = state["next_chapters"][1:]
    past_chapter = [current_chapter]
    new_summary = [agent_response["messages"][-1].content]
    
    logger.info(f"\nTask: {task_str}")
    logger.info(f"New summary: {new_summary[0]}")
    
    EVALUATION_PROMPT = """
    QUESTION: 

    Is the new summary similar or contained in the previous summary?

    SUMMARY SO FAR:

    {old}

    NEW SUMMARY:
    {new}

    """
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
    return {
        "next_chapters": remaining_chapters,
        "past_chapters": past_chapter,
        "summary": new_summary,
        "active_agent": "planner",
        "processing_complete": False
    }

# Initialize vector store
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

# Initialize language models
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

# Define workflow
workflow = StateGraph(PipelineState)

# Add nodes
workflow.add_node("planner", plan_step)
workflow.add_node("summarizer", summarize_step)

# Add edges
workflow.add_edge(START, "planner")
workflow.add_edge("summarizer", "planner")
workflow.add_edge("planner", "summarizer")
# Add conditional edges for completion
workflow.add_conditional_edges("planner", should_end, ["summarizer", END])

# Compile workflow
app = workflow.compile()

async def run_pipeline(query: str, file_paths: List[str]):
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
    
    print("Running pipeline...")
    summaries = []
    try:
        async for event in app.astream(initial_state, config={"recursion_limit": 50}):
            if "__end__" not in event:
                if "summary" in event and event["summary"]:
                    summaries.extend(event["summary"])
                    print(f"Chapter summary: {event['summary'][-1]}")
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
    return summaries

if __name__ == "__main__":
    query = """
    Please summarize the first three chapters from the book.
    """
    file_paths = [
        "/home/dusoudeth/Calibre Library/Joao Ubaldo Ribeiro/Viva o povo brasileiro (249)/Viva o povo brasileiro - Joao Ubaldo Ribeiro.pdf"
    ]
    
    summaries = asyncio.run(run_pipeline(query, file_paths))
    print("\nFinal Summaries:")
    for i, summary in enumerate(summaries, 1):
        print(f"\nChapter {i}:")
        print(summary)