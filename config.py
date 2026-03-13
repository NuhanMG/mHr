"""
Mobitel HR Assistant - Configuration
Centralized configuration settings for the RAG system.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

# --- Base Paths ---
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"
LOG_DIR = BASE_DIR / "chat_logs"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)

# --- Model Configuration ---
LLM_MODEL = "qwen2.5:7b"
LLM_TEMPERATURE = 0.2
EMBEDDING_MODEL = "nomic-embed-text:latest"

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4.1-nano"

# --- Active Model Provider ---
# "ollama" or "openai"
ACTIVE_LLM_PROVIDER = "ollama"

# --- Retrieval Configuration ---
RETRIEVAL_CONFIG = {
    "search_type": "mmr",  # Use MMR for diversity
    "k": 8,                # Number of final documents to return
    "fetch_k": 20,         # Number of documents to fetch before reranking
    "lambda_mult": 0.7,    # Diversity parameter (0=max diversity, 1=max relevance)
}

# --- Reranking Configuration ---
RERANKING_ENABLED = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5  # Number of documents after reranking

# --- Chunking Configuration ---
CHUNKING_CONFIG = {
    "default": {"chunk_size": 1000, "chunk_overlap": 200},
    "policy": {"chunk_size": 1200, "chunk_overlap": 300},
    "form": {"chunk_size": 500, "chunk_overlap": 100},
    "faq": {"chunk_size": 500, "chunk_overlap": 50},
    "manual": {"chunk_size": 1200, "chunk_overlap": 300},
}

# --- Input Validation ---
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 2000

# --- Rate Limiting ---
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# --- Caching ---
CACHE_ENABLED = True
CACHE_MAX_SIZE = 100
CACHE_TTL_SECONDS = 3600  # 1 hour

# --- Holiday / Leave Optimizer ---
HOLIDAYS_JSON_PATH = BASE_DIR / "holidays.json"

# --- Confidence Scoring ---
LOW_CONFIDENCE_THRESHOLD = 0.3
MEDIUM_CONFIDENCE_THRESHOLD = 0.6

# --- Logging Configuration ---
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.DEBUG  # Set to INFO for production

def setup_logging(name: str = "hr_assistant") -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name for identification
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(LOG_LEVEL)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# --- Query Expansion Keywords ---
# Maps informal terms to formal HR terminology
QUERY_EXPANSION_MAP = {
    # Leave related
    "vacation": "annual leave",
    "time off": "leave",
    "day off": "leave",
    "break": "leave",
    "sick": "medical leave",
    "maternity": "maternity leave",
    "paternity": "paternity leave",
    
    # Salary related
    "money early": "salary advance",
    "early salary": "salary advance",
    "advance payment": "salary advance",
    "bonus": "performance bonus",
    "pay": "salary",
    "increment": "salary increment",
    
    # Employment related
    "quit": "resignation",
    "leaving job": "resignation",
    "resign": "resignation",
    "exit": "resignation",
    "rejoin": "ex-staff rejoining",
    
    # Other
    "card lost": "identity card replacement",
    "ID lost": "identity card replacement",
    "transfer": "transfer posting",
    "relocation": "transfer",
    "work from home": "remote work",
    "WFH": "remote work",
    "training": "training and development",
    "course": "training",
    "workshop": "training",
    "harassment": "sexual harassment policy",
    "discipline": "employee discipline",
    "travel": "travel allowance",
    "foreign travel": "foreign travel policy",
    "phone": "official phone policy",
    "mobile": "official phone policy",
}


# --- Document Categories ---
DOCUMENT_CATEGORIES = {
    "forms": "form",
    "policies": "policy",
    "manuals": "manual",
    "FAQ": "faq",
    "Holiday": "holiday",
    "others": "other",
}
