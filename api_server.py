"""
Mobitel HR Assistant - FastAPI Server
Exposes REST API endpoints for the React frontend.

Endpoints:
- POST /api/chat     — Send a message and get RAG response
- GET  /api/faqs     — Retrieve all FAQ categories
- GET  /api/download — Download a document file
"""

import os
import uuid
import uvicorn
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import setup_logging

# Setup logger
logger = setup_logging("hr_assistant.api")

# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = None


class FileInfo(BaseModel):
    path: str
    name: str


class ChatResponse(BaseModel):
    answer: str
    files: List[FileInfo] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list)
    session_id: str
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    response_time: Optional[float] = None


class FAQItem(BaseModel):
    question: str
    answer: str


class FAQCategory(BaseModel):
    category: str
    faqs: List[FAQItem]


# --- FastAPI App ---

app = FastAPI(
    title="Mobitel HR Assistant API",
    description="REST API for the Mobitel HR Chatbot",
    version="1.0.0",
)

# CORS - allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lazy imports to avoid heavy loading at module level ---
_backend_loaded = False


def _ensure_backend():
    """Lazily import backend to trigger system initialization."""
    global _backend_loaded
    if not _backend_loaded:
        logger.info("Loading backend (RAG system initialization)...")
        import backend  # noqa: F401 — triggers initialize_system()
        _backend_loaded = True


# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on server startup."""
    logger.info("=" * 60)
    logger.info("STARTING MOBITEL HR ASSISTANT API SERVER")
    logger.info("=" * 60)
    _ensure_backend()
    logger.info("API server ready.")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the RAG pipeline.
    """
    _ensure_backend()
    from backend import get_answer, save_chat_history, convert_gradio_history_to_langchain, get_model_status

    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"Chat request: '{request.message[:60]}...' (session: {session_id[:8]}...)")

    # Convert history to LangChain format
    history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
    lc_history = convert_gradio_history_to_langchain(history_dicts)

    # Get answer from RAG engine
    answer, file_paths, follow_ups = get_answer(request.message, lc_history, session_id)

    # Build file info list
    files = []
    for fp in file_paths:
        if os.path.exists(fp):
            files.append(FileInfo(path=fp, name=os.path.basename(fp)))

    # Save history (append current exchange)
    full_history = history_dicts + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": answer},
    ]
    save_chat_history(full_history, session_id)

    logger.info(f"Chat response: {len(answer)} chars, {len(files)} files")

    # Get model status for response
    status = get_model_status()

    return ChatResponse(
        answer=answer,
        files=files,
        follow_ups=follow_ups,
        session_id=session_id,
        model_provider=status["provider"],
        model_name=status["model"],
        response_time=status["last_response_time"],
    )


@app.get("/api/faqs", response_model=List[FAQCategory])
async def get_faqs():
    """Return all FAQ categories with their Q&A pairs."""
    from faq_data import get_all_faqs

    faq_data = get_all_faqs()
    result = []
    for category, faqs in faq_data.items():
        items = [FAQItem(question=q, answer=a) for q, a in faqs]
        result.append(FAQCategory(category=category, faqs=items))

    return result


@app.get("/api/download")
async def download_file(filepath: str = Query(..., description="Absolute path to the file")):
    """Download a document file."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    # Security: only allow files from the data directory
    base_dir = Path(__file__).parent / "data"
    try:
        resolved = Path(filepath).resolve()
        if not str(resolved).startswith(str(base_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")

    return FileResponse(
        path=str(resolved),
        filename=resolved.name,
        media_type="application/octet-stream",
    )


class ModelSwitchRequest(BaseModel):
    provider: str = Field(..., description="'ollama' or 'openai'")


@app.post("/api/model/switch")
async def switch_model_endpoint(request: ModelSwitchRequest):
    """Switch the active LLM provider."""
    _ensure_backend()
    from backend import switch_model
    result = switch_model(request.provider)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/model/status")
async def model_status_endpoint():
    """Get current model status and last response time."""
    _ensure_backend()
    from backend import get_model_status
    return get_model_status()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Mobitel HR Assistant API"}


# --- Run ---

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
