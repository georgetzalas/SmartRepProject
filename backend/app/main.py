from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from .utils import process_query, on_load
from contextlib import asynccontextmanager

# Lifespan event using async context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    manual_path = os.getenv("MANUAL_PATH", "/app/data/bmw_x1.pdf")
    await on_load(manual_path)
    app.state.ready = True
    yield

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

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    response = await process_query(request.message)
    return {"response": response}

@app.get("/status")
def status():
    """Return status of document loading."""
    return {"ready": getattr(app.state, "ready", False)}