import os
import asyncio
import operator
import tiktoken
from typing import Annotated, List, Tuple, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatMaritalk
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
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
# utility functions...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@tool
def tokens_from_string(string:str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def load_file_contents(file_paths: List[str]) -> dict:
    file_contents = {}
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
        try:
            file_extension = Path(path).suffix.lower()
            if file_extension == '.pdf':
                logger.info(f"Loading PDF file: {path}")
                loader = PyPDFLoader(path)
                documents = loader.load()
                file_contents[path] = "\n".join(doc.page_content for doc in documents)
            elif file_extension in ['.docx', '.doc']:
                logger.info(f"Loading Word document: {path}")
                loader = Docx2txtLoader(path)
                documents = loader.load()
                file_contents[path] = "\n".join(doc.page_content for doc in documents)
            else:  # Default to text file handling
                logger.info(f"Loading text file: {path}")
                loader = TextLoader(path)
                documents = loader.load()
                with open(path, 'r') as f:
                    file_contents[path] = f.read()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 2048,
                chunk_overlap = 25,
                length_function = len,
                is_separator_regex = False,
            )
            splits = text_splitter.split_documents(documents)
            
            # Add page numbers to metadata if not present
            for i, doc in enumerate(splits):
                if not hasattr(doc.metadata, 'page_number'):
                    doc.metadata['page_number'] = i
            
            # Calculate tokens per chunk
            amount_of_tokens_per_chunk = [tokens_from_string(doc.page_content) for doc in splits]
            logger.info(f"Tokens per chunk: {amount_of_tokens_per_chunk}")
            
            # Add documents to vector store
            vector_store.add_documents(splits)
            logger.success(f"successfully processed: {path}")
        except Exception as e:
            logger.error(f"error loading {path}: {str(e)}")
            continue
            
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
PERSIST_DIRECTORY = "../db/vector_store"
embeddings = OpenAIEmbeddings(
    # model = "text-embedding-3-large",
    # model = "text-embedding-3-small",
    model="text-embedding-ada-002",
    api_key = os.getenv("OPENAI_API_KEY")
)
vector_store = Chroma(
    persist_directory = PERSIST_DIRECTORY, 
    embedding_function = embeddings
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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant."""),
        ("placeholder", "{messages}")
    ]
)

agent_planner = create_react_agent(
    model = llm_gpt_4o_mini,
    tools = list(),
    state_modifier = prompt
)

agent_summarizer = create_react_agent(
    model = llm_sabia_3,
    tools = list(),
    state_modifier = prompt
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

class Response(BaseModel):
    message: str = Field(description="Response message")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# setting up the planning...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PLANNER_PROMPT = """Your task is to plan the book summarizing."""

async def plan_step(state: PipelineState):
    """Create development plan"""
    # logger.section("reasoning")
    # logger.dict(state, "state")
    
    if (len(state["next_chapters"]) == 0) and (len(state["past_chapters"]) == 0):
        # Get documents from vector store
        result = vector_store.get()
        
        # Extract documents with their metadata
        documents = []
        if isinstance(result, dict):
            # Handle case where documents are stored with embeddings and metadatas
            for i, (doc, metadata) in enumerate(zip(result.get('documents', []), result.get('metadatas', []))):
                # Create a default page number if not present
                page_number = metadata.get('page_number', i) if metadata else i
                documents.append({
                    'page_number': page_number,
                    'content': doc
                })
        
        # Sort documents by page number
        sorted_documents = sorted(documents, key=lambda x: x['page_number'])
        
        # Create initial words list
        initial_words = []
        for doc in sorted_documents:
            preview = doc['content'][:150] if doc['content'] else ''
            initial_words.append((
                doc['page_number'],
                preview
            ))
        
        return {
            "next_chapters": initial_words,
            "active_agent": "code_agent"
        }
    elif len(state["past_chapters"]) > 0:
        return {
            "active_agent": "code_agent"
        }


def should_end(state: PipelineState):
    if (len(state["next_chapters"]) == 0) and (len(state["past_chapters"]) > 0): 
        return END
    else:
        return "code_agent"


async def code_agent_step(state: PipelineState):
    """Generate Next.js/TypeScript code"""
    # logger.section("acting")
    # logger.dict(state, "state")
    
    # Get the current chapter
    current_chapter = state["next_chapters"][0]
    current_chapter_page_number = current_chapter[0]
    current_chapter_str = current_chapter[1]
    
    # Perform RAG query
    relevant_docs = vector_store.similarity_search(current_chapter_str, k=1)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    
    # Create the task string
    task_str = f"""
    Please summarize the chapter from the book "Viva o povo brasileiro" by João Ubaldo Ribeiro, in Portuguese, given the following context:
    {context}

    SUMMARY SO FAR:
    {' '.join(state['summary'])}
    """
    # logger.info(task_str)
    
    # Get agent response
    agent_response = await agent_planner.ainvoke({
        "messages": [("user", task_str)]
    })
    
    # Prepare state updates
    # Remove the first chapter from next_chapters (we just processed it)
    remaining_chapters = state["next_chapters"][1:]
    
    # Add the current chapter to past_chapters as a list
    past_chapter = [current_chapter]
    
    # Add the new summary
    new_summary = [agent_response["messages"][-1].content]
    logger.info(f"New summary: {new_summary}")
    return {
        "next_chapters": remaining_chapters,
        "past_chapters": past_chapter,
        "summary": new_summary,
        "active_agent": "planner",
    }

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the workflow...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
workflow = StateGraph(PipelineState)
# adding nodes... 
workflow.add_node("planner", plan_step)
workflow.add_node("code_agent", code_agent_step)
# adding edges...
workflow.add_edge(START, "planner")
workflow.add_edge("code_agent", "planner")
# adding conditional edges...
workflow.add_conditional_edges("planner", should_end, ["code_agent", END])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# compiling the workflow and displaying the graph...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
app = workflow.compile()

display_graph(app)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# setting up the pipeline...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
async def run_pipeline(query: str, file_paths: List[str]):
    """
    Run the development pipeline
    """
    # loading file contents...
    print(f"loading file contents from: {file_paths}")
    _ = load_file_contents(file_paths)
    # defining the initial state...
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_chapters": list(),
        "past_chapters": list(),
        "summary": list(),
        "active_agent": "planner"
    }
    print("running pipeline...")
    async for event in app.astream(initial_state, config={"recursion_limit": 50}):
        if "__end__" not in event:
            # if "messages" in event:
            #     print(f"Response: {event['messages'][-1].content}")
            # if "current_plan" in event:
            #     print(f"Plan: {event['current_plan']}")
            pass


if __name__ == "__main__":
    query = """
    Please summarize the first three chapters from the book "Viva o povo brasileiro" by João Ubaldo Ribeiro.
    """
    file_paths = [
        "/home/dusoudeth/Calibre Library/Joao Ubaldo Ribeiro/Viva o povo brasileiro (249)/Viva o povo brasileiro - Joao Ubaldo Ribeiro.pdf"
    ]
    asyncio.run(run_pipeline(query, file_paths))