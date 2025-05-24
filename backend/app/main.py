from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from .utils import setup_rag_pipeline, process_query_with_rag
from contextlib import asynccontextmanager

manual_path = os.getenv("MANUAL_PATH", "./data/bmw_x1.pdf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.llm = None
    app.state.vector_store = None
    print("Application startup: Initializing RAG pipeline...")
    manual_path = os.getenv("MANUAL_PATH", "/app/data/bmw_x1.pdf")

    if not os.path.exists(manual_path):
        print(f"Error: Manual PDF not found at {manual_path}. Please ensure the file exists.")
    else:
        try:
            llm, vector_store = await setup_rag_pipeline(manual_path)
            app.state.llm = llm
            app.state.vector_store = vector_store
            app.state.ready = True
            print("RAG pipeline initialized successfully. Application is ready.")
        except Exception as e:
            print(f"Error during RAG pipeline initialization: {e}")
    yield
    print("Application shutdown: Cleaning up resources.")
    app.state.llm = None
    app.state.vector_store = None
    app.state.ready = False

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    page_references: list[int] = []

@app.get("/")
def root():
    return {"message": "API is running. Use /chat endpoint for queries."}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not getattr(app.state, 'ready', False) or not app.state.llm or not app.state.vector_store:
        raise HTTPException(status_code=503, detail="System not ready. Please try again later. Ensure the manual PDF was loaded correctly.")

    response_content, page_numbers = await process_query_with_rag(request.message, app.state.llm, app.state.vector_store)
    return {"response": response_content, "page_references": page_numbers}

@app.get("/status")
def status():
    is_ready = getattr(app.state, "ready", False)
    llm_loaded = app.state.llm is not None
    vector_store_loaded = app.state.vector_store is not None
    return {
        "ready": is_ready,
        "llm_initialized": llm_loaded,
        "vector_store_initialized": vector_store_loaded
    }
