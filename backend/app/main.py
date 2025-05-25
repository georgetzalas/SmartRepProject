from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
import logging
import json
import os
from .utils import setup_rag_pipeline, process_query_with_rag # Updated import names
from contextlib import asynccontextmanager

# Configure logger
logger = logging.getLogger("uvicorn.error")

manual_path = os.getenv("MANUAL_PATH", "./data/bmw_x1.pdf") # Changed default

# Lifespan event using async context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    Loads the RAG pipeline components (LLM and vector store) on startup.
    """
    app.state.ready = False
    app.state.llm = None
    app.state.vector_store = None
    app.state.memory = None  # Initialize memory to None or appropriate value
    app.state._state = None  # Initialize _state to None or appropriate value
    print("Application startup: Initializing RAG pipeline...")
    manual_path = os.getenv("MANUAL_PATH", "/app/data/bmw_x1.pdf") # Ensure this PDF is available at this path

    if not os.path.exists(manual_path):
        print(f"Error: Manual PDF not found at {manual_path}. Please ensure the file exists.")
        # You might want to raise an exception here or handle it differently
        # For now, the app will start but /chat endpoint will fail gracefully
    else:
        try:
            llm, vector_store, memory = await setup_rag_pipeline(manual_path)
            app.state.llm = llm
            app.state.vector_store = vector_store
            app.state.memory = memory  # Initialize memory as None or set appropriately
            app.state.ready = True
            print("RAG pipeline initialized successfully. Application is ready.")
        except Exception as e:
            print(f"Error during RAG pipeline initialization: {e}")
            # App will start, but ready will be false.
    yield
    # Clean up resources if any (e.g., app.state.llm = None, app.state.vector_store = None)
    print("Application shutdown: Cleaning up resources.")
    app.state.llm = None
    app.state.vector_store = None
    app.state.memory = None
    app.state.ready = False


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    page_references: list[int] = [] # To include page numbers
    relevant_images: list[str] = []  # To include image paths

# Compute the absolute path to the data/images directory:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(BASE_DIR, "data", "images")

# Add static file mounting for images here
app.mount("/images", StaticFiles(directory=images_dir), name="images")

@app.get("/api/images")
def get_images(pages: str = Query(..., description="Comma-separated list of page numbers")):
    """
    Given comma-separated page numbers, return all related image paths.
    Assumes image_database.json is in the app folder with a mapping of page numbers to a list of image file paths.
    """
    json_path = os.path.join(os.path.dirname(__file__), "image_database.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Image database not found")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image database: {e}")

    # Convert input pages to list of ints
    try:
        requested_pages = [str(int(p.strip())) for p in pages.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid page number format")

    # Gather image paths for requested pages
    images = []
    for page in requested_pages:
        # Our JSON keys can be either string or number;
        # ensure we use string keys. If page not found, ignore it.
        page_images = data.get(page, [])
        images.extend(page_images)
    
    return {"images": images}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Handles chat requests. Processes the user's query using the RAG pipeline.
    """
    if not getattr(app.state, 'ready', False) or not app.state.llm or not app.state.vector_store:
        raise HTTPException(status_code=503, detail="System not ready. Please try again later. Ensure the manual PDF was loaded correctly.")

    if not app.state.memory:
        logger.info("First request: Initializing new conversation memory")
        app.state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        app.state.memory.chat_memory.messages.append(
            SystemMessage(content="You are a helpful BMW X1 assistant. Answer questions based strictly on the manual content.")
        )
    
    response, page_references = await process_query_with_rag(
        request.message,
        app.state.llm,
        app.state.vector_store,
        app.state.memory
    )
    return {
        "response": response, 
        "page_references": page_references,
    }

@app.get("/status")
def status():
    """Return status of document loading."""
    is_ready = getattr(app.state, "ready", False)
    llm_loaded = app.state.llm is not None
    vector_store_loaded = app.state.vector_store is not None
    memory_loaded = app.state.memory is not None
    return {
        "ready": is_ready,
        "llm_initialized": llm_loaded,
        "vector_store_initialized": vector_store_loaded,
        "memory_initialized": memory_loaded
        }