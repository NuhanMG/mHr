"""
Mobitel HR Assistant - Backend (RAG Logic)
This module contains all the RAG (Retrieval Augmented Generation) logic.

Features:
- MMR retrieval for diverse results
- Cross-encoder reranking for better relevance
- Structured prompt templates
- Input validation and sanitization
- Query expansion for non-tech users
- Answer confidence scoring
- Comprehensive logging
- Caching for frequent queries
"""

import os
import re
import json
import time
from datetime import datetime
from typing import Optional, Tuple, List

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Import configuration
from config import (
    setup_logging,
    BASE_DIR,
    VECTORSTORE_PATH,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    LOG_DIR,
    RETRIEVAL_CONFIG,
    RERANKING_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    CACHE_ENABLED,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ACTIVE_LLM_PROVIDER,
)

# Import utilities
from utils import (
    validate_input,
    expand_query,
    rate_limiter,
    query_cache,
    calculate_confidence,
    format_low_confidence_answer,
)

# Import leave optimizer
from leave_optimizer import handle_leave_query

# Import hardcoded FAQ search
from faq_data import search_faqs

# Setup loggers for different components
logger = setup_logging("hr_assistant.backend")
retrieval_logger = setup_logging("hr_assistant.retrieval")
llm_logger = setup_logging("hr_assistant.llm")

# --- Global Components ---
vectorstore = None
rag_chain = None
rag_chain_openai = None
reranker = None

# --- Model State ---
current_provider = ACTIVE_LLM_PROVIDER  # "ollama" or "openai"
last_response_time = None  # seconds for last answer


# ============================================================================
# STRUCTURED PROMPT TEMPLATES
# ============================================================================

# Contextualize question prompt
CONTEXTUALIZE_Q_PROMPT = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

# Base QA system prompt with HR-specific enhancements
QA_SYSTEM_PROMPT = """You are a friendly and professional HR Assistant for Mobitel Sri Lanka.

ANSWER LENGTH — ADAPTIVE:
- For SIMPLE questions (definition, yes/no, factual): Give a concise answer in 2-3 sentences.
- For PROCEDURAL questions (steps, processes, "how do I", applications): Give a DETAILED answer with numbered steps, required documents, contact persons, timelines, and relevant forms. Be thorough — employees rely on this.
- For POLICY questions: Summarize the key points clearly, mention any important exceptions or eligibility criteria.
- Always finish your answer completely. Never trail off or leave sentences incomplete.

CRITICAL RULES:
1. Use simple, friendly language — employees are not HR experts.
2. Use bullet points or numbered lists when there are 3+ points.
3. If the information is NOT available in the provided context, you MUST say exactly: "I don't have knowledge on that scenario. Please contact the HR department directly for assistance."
4. Never fabricate or guess information. Only use what is in the context.
5. When mentioning forms or documents, always name them clearly.

FILE HANDLING (VERY IMPORTANT):
- If your answer mentions ANY form, application, policy document, or downloadable file, you MUST include its path.
- At the END of your response (BEFORE the follow-up questions), include [[FILE:exact/path/from/Source/field]] for EACH relevant document.
- You can include MULTIPLE [[FILE:...]] tags if multiple documents are relevant.
- Use the EXACT path from the 'Source' field in the context.
- ONLY include documents that are DIRECTLY relevant to the specific question asked. Do NOT include documents about unrelated topics. For example, do NOT include health insurance documents when the question is about salary advance or travel.
- NEVER write raw file paths (like C:\\Users\\... or /home/...) in the answer body. Only place file paths inside [[FILE:...]] tags. In the answer text, refer to documents by their name only (e.g., "Salary Advance Form" not the full path).

FOLLOW-UP QUESTIONS (VERY IMPORTANT):
- At the very END of your response, ALWAYS suggest 2-3 related follow-up questions the employee might want to ask next.
- Format each as: [[FOLLOWUP:question text here]]
- These should be natural, relevant questions that logically follow from your answer.
- Examples: if answering about leave policy, suggest asking about leave application process, leave balance, etc.

COMMON TERMS:
- "vacation", "time off" = annual leave
- "sick" = medical leave  
- "early salary" = salary advance
- "ID lost" = identity card replacement

CONTEXT:
{context}"""

# Prompt for form-specific requests
FORM_REQUEST_PROMPT = """You are helping a Mobitel employee download an HR form.

Based on the context, identify the EXACT form they need and provide:
1. The form name
2. A brief description of when to use this form (1-2 sentences)
3. At the END, include: [[FILE:exact/path/from/context]]

Be concise and helpful. Use the exact file path from the Source field.

CONTEXT:
{context}"""

# Prompt for policy queries
POLICY_QUERY_PROMPT = """You are explaining HR policies to a Mobitel employee.

Based on the context, provide:
1. A clear, simple explanation in plain language
2. Key points as bullet points (if applicable)
3. Any important exceptions or conditions
4. Who to contact for more details (if mentioned)

Keep the response professional but friendly. Avoid HR jargon.

CONTEXT:
{context}"""


# ============================================================================
# RERANKER
# ==========

class DocumentReranker:
    """Cross-encoder based document reranker for improved relevance."""
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = None
        self.logger = setup_logging("hr_assistant.reranker")
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.logger.info("Reranker model loaded successfully")
        except ImportError:
            self.logger.warning("sentence-transformers not installed. Reranking disabled.")
            self.model = None
        except Exception as e:
            self.logger.error(f"Failed to load reranker: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Document], top_k: int = RERANKER_TOP_K) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if not self.model or not documents:
            return documents[:top_k]
        
        self.logger.debug(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
        
        try:
            # Create query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Sort by score (descending)
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Log top scores
            for i, (doc, score) in enumerate(doc_scores[:3]):
                source = doc.metadata.get('source', 'Unknown')[:50]
                self.logger.debug(f"  Rank {i+1}: score={score:.4f}, source={source}...")
            
            reranked = [doc for doc, score in doc_scores[:top_k]]
            self.logger.info(f"Reranking complete. Returned top {len(reranked)} documents.")
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return documents[:top_k]


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_system() -> bool:
    """
    Initialize the RAG system with vectorstore and chains.

    Returns:
        True if initialization successful, False otherwise
    """
    global vectorstore, rag_chain, rag_chain_openai, reranker

    logger.info("="*60)
    logger.info("INITIALIZING HR ASSISTANT RAG SYSTEM")
    logger.info("="*60)

    # Step 1: Load main vectorstore
    logger.info("Step 1/5: Loading main vectorstore...")
    if os.path.exists(str(VECTORSTORE_PATH)):
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(
                str(VECTORSTORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"  ✓ Main vectorstore loaded from {VECTORSTORE_PATH}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load main vectorstore: {e}")
            return False
    else:
        logger.error(f"  ✗ Main vectorstore not found at {VECTORSTORE_PATH}")
        logger.error("  → Please run PRE/ingest.py first to create the vectorstore")
        return False


    # Step 3: Initialize LLM (Ollama)
    logger.info("Step 3/5: Initializing Ollama LLM...")
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        logger.info(f"  ✓ Ollama LLM initialized: {LLM_MODEL} (temp={LLM_TEMPERATURE})")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize Ollama LLM: {e}")
        return False

    # Step 3b: Initialize OpenAI LLM (if key available)
    llm_openai = None
    if OPENAI_API_KEY:
        logger.info("Step 3b: Initializing OpenAI LLM...")
        try:
            from langchain_openai import ChatOpenAI
            llm_openai = ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=LLM_TEMPERATURE,
                api_key=OPENAI_API_KEY,
            )
            logger.info(f"  ✓ OpenAI LLM initialized: {OPENAI_MODEL}")
        except Exception as e:
            logger.warning(f"  ⚠ Failed to initialize OpenAI LLM: {e}")
    else:
        logger.info("  → No OpenAI API key found, skipping OpenAI initialization")

    # Step 4: Setup retriever with MMR
    logger.info("Step 4/5: Setting up MMR retriever...")
    retriever = vectorstore.as_retriever(
        search_type=RETRIEVAL_CONFIG["search_type"],
        search_kwargs={
            "k": RETRIEVAL_CONFIG["k"],
            "fetch_k": RETRIEVAL_CONFIG["fetch_k"],
            "lambda_mult": RETRIEVAL_CONFIG["lambda_mult"],
        }
    )
    logger.info(f"  ✓ MMR Retriever configured: k={RETRIEVAL_CONFIG['k']}, fetch_k={RETRIEVAL_CONFIG['fetch_k']}")

    # Step 5: Build RAG chains
    logger.info("Step 5/5: Building RAG chains...")

    # Contextualize question chain
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # QA chain with structured prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Document formatting includes source
    document_prompt = ChatPromptTemplate.from_template(
        "Content: {page_content}\nSource: {source}"
    )

    # Build Ollama RAG chain
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=document_prompt
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    logger.info("  ✓ Ollama RAG chain built successfully")

    # Build OpenAI RAG chain (if available)
    if llm_openai:
        history_aware_retriever_openai = create_history_aware_retriever(
            llm_openai, retriever, contextualize_q_prompt
        )
        question_answer_chain_openai = create_stuff_documents_chain(
            llm_openai, qa_prompt, document_prompt=document_prompt
        )
        rag_chain_openai = create_retrieval_chain(history_aware_retriever_openai, question_answer_chain_openai)
        logger.info("  ✓ OpenAI RAG chain built successfully")

    # Step 6: Initialize reranker (optional)
    if RERANKING_ENABLED:
        logger.info("Step 6 : Initializing document reranker...")
        reranker = DocumentReranker()
        if reranker.model:
            logger.info("  ✓ Reranker initialized")
        else:
            logger.warning("  ⚠ Reranker not available, continuing without it")

    logger.info("="*60)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("="*60)

    return True


# ============================================================================
# HOLIDAY QUERY INTERCEPTOR
# ============================================================================

# Month name mappings for holiday queries
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _load_holidays_json() -> dict:
    """Load holidays.json data. Returns None if not found."""
    holidays_path = os.path.join(str(BASE_DIR), "holidays.json")
    if not os.path.exists(holidays_path):
        return None
    try:
        import json
        with open(holidays_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _check_holiday_query(query: str) -> str:
    """
    Check if the query is about holidays and answer from structured JSON data.
    Returns an answer string if it's a holiday query, None otherwise.
    """
    query_lower = query.lower().strip()
    
    # Skip if this is a LEAVE PLANNING query (let leave_optimizer handle it)
    leave_planning_keywords = [
        "suggest", "recommend", "plan", "suitable", "best day", "best time",
        "want to get", "want to take", "days off", "how to maximize",
        "optimal", "when should i take", "when to take", "give me"
    ]
    if any(kw in query_lower for kw in leave_planning_keywords):
        return None
    
    # Holiday-related keywords
    holiday_keywords = ["holiday", "holidays", "poya", "public holiday", "company holiday",
                        "bank holiday", "day off"]
    
    # Check if this is a holiday query
    is_holiday_query = any(kw in query_lower for kw in holiday_keywords)
    
    # Also check for "when is [specific holiday]" patterns
    when_pattern = bool(re.search(r'\b(when|what date|which date)\b.*\b(is|are|does)\b', query_lower))
    specific_holiday_names = ["vesak", "pongal", "christmas", "independence", "new year",
                               "good friday", "poson", "esala", "hadji", "milad", "nikini",
                               "medin", "bak", "unduvap", "il full moon"]
    mentions_specific = any(name in query_lower for name in specific_holiday_names)
    
    if not is_holiday_query and not (when_pattern and mentions_specific):
        return None
    
    # Load holiday data
    data = _load_holidays_json()
    if not data:
        return None  # Fall through to RAG
    
    year = data.get("year", 2026)
    holidays = data.get("holidays", [])
    
    if not holidays:
        return None
    
    # Detect month-specific query
    target_month = None
    for month_name, month_num in _MONTH_NAMES.items():
        if month_name in query_lower:
            target_month = month_num
            break
    
    # Case 1: Month-specific holiday query
    if target_month:
        from datetime import datetime
        month_name_full = datetime(year, target_month, 1).strftime("%B")
        month_holidays = [h for h in holidays 
                          if int(h["date"].split("-")[1]) == target_month]
        
        if month_holidays:
            lines = [f"📅 **Holidays in {month_name_full} {year}:**\n"]
            for h in month_holidays:
                d = datetime.strptime(h["date"], "%Y-%m-%d")
                lines.append(f"- **{d.strftime('%d %B')}** ({d.strftime('%A')}) — {h['name']}")
            lines.append(f"\nThere {'is' if len(month_holidays) == 1 else 'are'} **{len(month_holidays)}** holiday{'s' if len(month_holidays) != 1 else ''} in {month_name_full}.")
            return "\n".join(lines)
        else:
            return f"There are **no public holidays** in {month_name_full} {year}. The nearest holidays would be in other months. You can ask me to list all holidays for the year."
    
    # Case 2: Specific holiday name query
    if mentions_specific:
        from datetime import datetime
        matching = []
        for h in holidays:
            if any(name in h["name"].lower() for name in specific_holiday_names if name in query_lower):
                matching.append(h)
        
        if matching:
            lines = []
            for h in matching:
                d = datetime.strptime(h["date"], "%Y-%m-%d")
                lines.append(f"📅 **{h['name']}** is on **{d.strftime('%d %B %Y')}** ({d.strftime('%A')}).")
            return "\n".join(lines)
    
    # Case 3: General "list/all holidays" query
    if any(word in query_lower for word in ["all", "list", "full", "every", "how many"]):
        from datetime import datetime
        lines = [f"📅 **Company Holidays for {year}** ({len(holidays)} holidays):\n"]
        
        current_month = None
        for h in holidays:
            d = datetime.strptime(h["date"], "%Y-%m-%d")
            month_label = d.strftime("%B")
            if month_label != current_month:
                current_month = month_label
                lines.append(f"\n**{month_label}:**")
            lines.append(f"- {d.strftime('%d')} ({d.strftime('%A')}) — {h['name']}")
        
        return "\n".join(lines)
    
    # Case 4: General holiday question — provide a helpful summary
    if is_holiday_query:
        from datetime import datetime
        today = datetime.now().date()
        
        # Find next upcoming holiday
        upcoming = []
        for h in holidays:
            from datetime import date as date_type
            parts = h["date"].split("-")
            hd = date_type(int(parts[0]), int(parts[1]), int(parts[2]))
            if hd >= today:
                upcoming.append((hd, h["name"]))
        
        answer = f"There are **{len(holidays)} public holidays** in {year}."
        
        if upcoming:
            next_date, next_name = upcoming[0]
            days_until = (next_date - today).days
            answer += f"\n\nThe next upcoming holiday is **{next_name}** on **{next_date.strftime('%d %B %Y')}** ({next_date.strftime('%A')})"
            if days_until == 0:
                answer += " — that's today! 🎉"
            elif days_until == 1:
                answer += " — that's tomorrow!"
            else:
                answer += f" — **{days_until} days** from now."
        
        answer += "\n\nWould you like me to list all holidays for the year, or holidays in a specific month?"
        return answer
    
    return None


# ============================================================================
# MODEL SWITCHING
# ============================================================================

def switch_model(provider: str) -> dict:
    """Switch the active LLM provider between 'ollama' and 'openai'."""
    global current_provider
    if provider not in ("ollama", "openai"):
        return {"success": False, "error": f"Unknown provider: {provider}"}
    if provider == "openai" and rag_chain_openai is None:
        return {"success": False, "error": "OpenAI model not initialized. Check API key."}
    current_provider = provider
    logger.info(f"Switched LLM provider to: {provider}")
    return {"success": True, "provider": provider}


def get_model_status() -> dict:
    """Return current model info and last response time."""
    provider = current_provider
    model_name = OPENAI_MODEL if provider == "openai" else LLM_MODEL
    return {
        "provider": provider,
        "model": model_name,
        "last_response_time": last_response_time,
        "openai_available": rag_chain_openai is not None,
    }


# ============================================================================
# MAIN ANSWER FUNCTION
# ============================================================================

def get_answer(message: str, history: List, session_id: str = "default") -> Tuple[str, List[str], List[str]]:
    """
    Get answer from RAG Chain with all improvements.
    
    Args:
        message: User's question as a string
        history: List of LangChain messages (HumanMessage, AIMessage)
        session_id: Session identifier for rate limiting
    
    Returns:
        tuple: (answer_string, list_of_file_paths, list_of_follow_up_questions)
    """
    logger.info("-"*60)
    logger.info(f"NEW QUERY: '{message[:100]}...' (session: {session_id[:8]}...)")
    logger.info(f"  Using provider: {current_provider}")
    logger.info("-"*60)

    start_time = time.time()
    
    # Phase 1: Input Validation
    logger.info("Phase 1: Input Validation")
    is_valid, sanitized_message, error_msg = validate_input(message)
    if not is_valid:
        logger.warning(f"  ✗ Validation failed: {error_msg}")
        return error_msg, [], []
    logger.info(f"  ✓ Input validated and sanitized")
    
    # Phase 2: Rate Limiting
    logger.info("Phase 2: Rate Limiting Check")
    is_allowed, rate_msg = rate_limiter.is_allowed(session_id)
    if not is_allowed:
        logger.warning(f"  ✗ Rate limited: {rate_msg}")
        return rate_msg, [], []
    logger.info("  ✓ Rate limit check passed")
    
    # Phase 3: Cache Check
    if CACHE_ENABLED:
        logger.info("Phase 3: Cache Check")
        history_hash = str(len(history))  # Simple hash for demo
        cached_result = query_cache.get(sanitized_message, history_hash)
        if cached_result:
            logger.info("  ✓ Cache hit! Returning cached result")
            elapsed = round(time.time() - start_time, 2)
            _update_response_time(elapsed)
            return cached_result
        logger.info("  → Cache miss, proceeding with retrieval")
    
    # Phase 4: Query Expansion
    logger.info("Phase 4: Query Expansion")
    expanded_query = expand_query(sanitized_message)
    if expanded_query != sanitized_message:
        logger.info(f"  ✓ Query expanded: '{expanded_query[:80]}...'")
    else:
        logger.info("  → No expansion needed")
    
    # Phase 4.3: Holiday Query Interceptor (answers from structured JSON, not RAG)
    logger.info("Phase 4.3: Holiday Query Check")
    holiday_answer = _check_holiday_query(sanitized_message)
    if holiday_answer:
        logger.info("  ✓ Holiday question answered from structured data")
        elapsed = round(time.time() - start_time, 2)
        _update_response_time(elapsed)
        if CACHE_ENABLED:
            query_cache.set(sanitized_message, (holiday_answer, [], []), history_hash)
        return holiday_answer, [], []
    logger.info("  → Not a holiday-specific query")
    
    # Phase 4.5: Leave Suggestion Check
    logger.info("Phase 4.5: Leave Intent Detection")
    leave_result = handle_leave_query(sanitized_message)
    if leave_result:
        logger.info("  ✓ Leave suggestion generated! Returning directly.")
        elapsed = round(time.time() - start_time, 2)
        _update_response_time(elapsed)
        if CACHE_ENABLED:
            query_cache.set(sanitized_message, (leave_result, [], []), history_hash)
        return leave_result, [], []
    logger.info("  → Not a leave suggestion query.")
    
    # Phase 4.6: Hardcoded FAQ Check
    logger.info("Phase 4.6: Hardcoded FAQ Exact Match")
    faq_exact = search_faqs(sanitized_message)
    if faq_exact:
        logger.info("  ✓ Hardcoded FAQ match found! Returning directly.")
        elapsed = round(time.time() - start_time, 2)
        _update_response_time(elapsed)
        if CACHE_ENABLED:
            query_cache.set(sanitized_message, (faq_exact, [], []), history_hash)
        return faq_exact, [], []
    logger.info("  → No hardcoded FAQ match.")
    
    # Phase 5: Proceeding to RAG
    logger.info("Phase 5: No FAQ match. Proceeding with full RAG chain.")
    
    # Phase 6: RAG Chain Execution
    logger.info("Phase 6: RAG Chain Execution")
    active_chain = rag_chain_openai if current_provider == "openai" and rag_chain_openai else rag_chain
    if active_chain is None:
        error = "⚠️ System not initialized. Please contact the administrator."
        logger.error(f"  ✗ RAG chain not initialized")
        return error, [], []

    logger.info(f"  Using {'OpenAI' if active_chain is rag_chain_openai else 'Ollama'} chain")
    
    file_paths = []
    follow_ups = []
    answer = ""
    source_docs = []
    
    try:
        # Invoke RAG chain
        llm_logger.info(f"Invoking RAG chain with query: '{expanded_query[:50]}...'")
        response = active_chain.invoke({
            "input": expanded_query, 
            "chat_history": history
        })
        
        answer = response.get("answer", "")
        source_docs = response.get("context", [])
        
        retrieval_logger.info(f"Retrieved {len(source_docs)} documents")
        for i, doc in enumerate(source_docs[:3]):
            source = doc.metadata.get('source', 'Unknown')
            retrieval_logger.debug(f"  Doc {i+1}: {source}")
        
        llm_logger.info(f"LLM response length: {len(answer)} chars")
        llm_logger.debug(f"LLM response preview: '{answer[:100]}...'")
        
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"  ✗ RAG chain error ({error_type}): {e}")
        
        # User-friendly error messages
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            return "⚠️ The AI service is temporarily unavailable. Please try again in a few moments.", [], []
        elif "timeout" in str(e).lower():
            return "⚠️ The request took too long. Please try a shorter question.", [], []
        else:
            return "⚠️ Something went wrong while processing your question. Please try again.", [], []
    
    # Phase 7: Answer Confidence Check
    logger.info("Phase 7: Answer Confidence Check")
    confidence_score, confidence_level = calculate_confidence(
        sanitized_message, answer, source_docs
    )
    logger.info(f"  → Confidence: {confidence_score:.2f} ({confidence_level})")
    
    # Add disclaimer for low confidence
    answer = format_low_confidence_answer(answer, confidence_level)
    
    # Phase 8: File Path Extraction (supports multiple files)
    logger.info("Phase 8: Multi-File Path Extraction")
    import re as re_module
    file_markers = re_module.findall(r'\[\[FILE:(.*?)\]\]', answer)
    if file_markers:
        logger.info(f"  → Found {len(file_markers)} file marker(s) in LLM response")
        for extracted_path in file_markers:
            extracted_path = extracted_path.strip()
            logger.debug(f"  → Checking path: {extracted_path}")
            
            # Try to find the file
            resolved = None
            if os.path.exists(extracted_path):
                resolved = extracted_path
            else:
                abs_path = os.path.abspath(extracted_path)
                if os.path.exists(abs_path):
                    resolved = abs_path
            
            if resolved and resolved not in file_paths:
                file_paths.append(resolved)
                logger.info(f"  ✓ File found: {resolved}")
            else:
                logger.warning(f"  ⚠ File not found: {extracted_path}")
            
            # Clean the marker from answer
            answer = answer.replace(f"[[FILE:{extracted_path}]]", "").strip()
    
    # Phase 8.5: Follow-up Question Extraction
    logger.info("Phase 8.5: Follow-up Question Extraction")
    followup_markers = re_module.findall(r'\[\[FOLLOWUP:(.*?)\]\]', answer)
    if followup_markers:
        follow_ups = [q.strip() for q in followup_markers if q.strip()]
        logger.info(f"  → Found {len(follow_ups)} follow-up question(s)")
        for fq in followup_markers:
            answer = answer.replace(f"[[FOLLOWUP:{fq}]]", "").strip()
    else:
        logger.info("  → No follow-up questions generated")
    
    # Phase 8.7: Sanitize file paths from answer text (security)
    # The LLM sometimes embeds absolute file paths in the answer body (outside [[FILE:]] markers).
    # These expose server directory structure and must be stripped before returning to the user.
    logger.info("Phase 8.7: Sanitizing file paths from answer")
    # Match Windows absolute paths (C:\...\something.pdf) and Unix paths (/home/.../something.pdf)
    answer = re.sub(
        r'(?:(?:[A-Za-z]:\\|/)[\w\\/.:\- ]+?[\\/])([^\\/\n]+\.pdf)',
        r'\1',
        answer
    )
    # Also remove any "located at:" / "found at:" / "path:" prefixes left orphaned
    answer = re.sub(r'\(?\s*(?:located|found|available|stored)\s+at:\s*([^)\n]+\.pdf)\s*\)?', r'(\1)', answer)

    # Phase 9: Smart document detection - search for forms AND policies
    doc_keywords = ["form", "application", "download", "submit", "fill", "registration",
                    "policy", "procedure", "guideline", "manual"]
    answer_mentions_doc = any(kw in answer.lower() for kw in doc_keywords)
    query_mentions_doc = any(kw in message.lower() for kw in doc_keywords)
    
    if not file_paths and (answer_mentions_doc or query_mentions_doc):
        logger.info("Phase 9: Smart Document Detection")
        # Use only the user's original query — NOT the full answer text.
        # Including the answer introduces hundreds of irrelevant terms that cause
        # cross-topic matches (e.g., health insurance docs for salary advance queries).
        found_files = find_matching_files(message, max_results=3)
        for f in found_files:
            if f not in file_paths:
                file_paths.append(f)
                logger.info(f"  ✓ Document found: {os.path.basename(f)}")
        if file_paths:
            download_names = ', '.join(os.path.basename(f) for f in file_paths)
            answer += f"\n\n📄 **Downloads available:** {download_names}"
    
    # Phase 10: Cache Result
    if CACHE_ENABLED and answer:
        logger.info("Phase 10: Caching Result")
        query_cache.set(sanitized_message, (answer, file_paths, follow_ups), history_hash)
        logger.info("  ✓ Result cached")
    
    logger.info("-"*60)
    logger.info(f"QUERY COMPLETE: Answer length={len(answer)}, Files={len(file_paths)}, Follow-ups={len(follow_ups)}")
    elapsed = round(time.time() - start_time, 2)
    _update_response_time(elapsed)
    logger.info(f"  Response time: {elapsed}s (provider: {current_provider})")
    logger.info("-"*60)

    return answer, file_paths, follow_ups


def _update_response_time(elapsed: float):
    """Update the global last response time."""
    global last_response_time
    last_response_time = elapsed


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_matching_files(query: str, max_results: int = 3) -> List[str]:
    """
    Searches for matching files in data directory based on query keywords.
    Returns multiple matches (forms AND policy documents).
    Uses weighted scoring that prioritizes domain-specific terms over generic ones.
    Only uses the user's original query for matching — NOT the full answer text.

    Args:
        query: User's original query string (should NOT include the LLM answer)
        max_results: Maximum number of files to return

    Returns:
        List of paths to matching files, sorted by relevance
    """
    retrieval_logger.debug(f"Searching for files matching: '{query[:80]}...'")

    query_lower = query.lower()

    # Generic/stop words that match too many files — EXCLUDED from scoring
    stop_words = {
        "how", "do", "i", "the", "a", "an", "to", "for", "of", "in", "on", "is", "it",
        "my", "can", "what", "where", "when", "who", "which", "this", "that", "are",
        "form", "apply", "application", "document", "documents", "download", "get",
        "need", "want", "please", "help", "about", "with", "from", "should", "would",
        "staff", "policy", "procedure",
    }

    # Filename stop words — generic terms that appear in filenames but carry no topic meaning
    filename_stop_words = {
        "faq", "final", "addendum", "user", "manual", "version",
        "2023", "2024", "2025", "2026", "2027",
        "v1", "v2", "v3", "v4", "v5",
    }

    # Topic groups: each group defines a coherent topic with its query triggers and file terms
    # This prevents cross-topic matches (e.g., salary query matching health insurance files)
    topic_groups = {
        "medical_insurance": {
            "query_triggers": {"medical", "insurance", "health", "hospital", "claim", "hospitalization"},
            "file_terms": {"medical", "insurance", "health", "hospital"},
        },
        "family_package": {
            "query_triggers": {"family", "package", "spouse", "children", "member", "registration", "inclusion", "newborn", "baby"},
            "file_terms": {"family", "package", "member", "registration", "inclusion"},
        },
        "salary_advance": {
            "query_triggers": {"salary", "advance", "pay"},
            "file_terms": {"salary", "advance"},
        },
        "bank_account": {
            "query_triggers": {"bank", "account"},
            "file_terms": {"bank", "account", "change"},
        },
        "travel_foreign": {
            "query_triggers": {"travel", "foreign", "overseas", "abroad"},
            "file_terms": {"travel", "foreign", "overseas"},
        },
        "travel_domestic": {
            "query_triggers": {"domestic", "travel", "meal", "allowance"},
            "file_terms": {"domestic", "travel", "meal", "allowance"},
        },
        "visa": {
            "query_triggers": {"visa", "business"},
            "file_terms": {"visa", "business"},
        },
        "phone_device": {
            "query_triggers": {"phone", "device", "official"},
            "file_terms": {"phone", "device", "official"},
        },
        "reimbursement": {
            "query_triggers": {"reimbursement", "fee", "membership"},
            "file_terms": {"reimbursement", "fee", "membership"},
        },
        "leave": {
            "query_triggers": {"leave", "vacation", "annual", "sick", "maternity", "paternity"},
            "file_terms": {"leave"},
        },
        "training": {
            "query_triggers": {"training", "development", "course"},
            "file_terms": {"training", "development", "course"},
        },
        "harassment": {
            "query_triggers": {"harassment", "sexual"},
            "file_terms": {"harassment", "sexual"},
        },
        "discipline": {
            "query_triggers": {"discipline", "disciplinary", "misconduct"},
            "file_terms": {"discipline"},
        },
        "bonding": {
            "query_triggers": {"bonding", "bond"},
            "file_terms": {"bonding", "bond"},
        },
        "identity": {
            "query_triggers": {"identity", "card", "id"},
            "file_terms": {"identity", "card"},
        },
        "rejoining": {
            "query_triggers": {"rejoining", "rejoin"},
            "file_terms": {"rejoining", "ex"},
        },
        "coop": {
            "query_triggers": {"coop", "cooperative"},
            "file_terms": {"coop"},
        },
        "peo": {
            "query_triggers": {"peo", "peotv"},
            "file_terms": {"peo"},
        },
        "migration": {
            "query_triggers": {"migration", "migrate"},
            "file_terms": {"migration"},
        },
        "interview": {
            "query_triggers": {"interview", "pin"},
            "file_terms": {"interview", "pin"},
        },
        "contact": {
            "query_triggers": {"contact", "number", "email", "reach"},
            "file_terms": {"contact"},
        },
    }

    # Extract meaningful terms from query only
    import string
    raw_terms = set(query_lower.split())
    cleaned_terms = {t.strip(string.punctuation) for t in raw_terms}
    meaningful_terms = {t for t in cleaned_terms if t not in stop_words and len(t) > 2}

    # Detect which topics the query belongs to
    active_topics = set()
    for topic_name, topic_info in topic_groups.items():
        if meaningful_terms & topic_info["query_triggers"]:
            active_topics.add(topic_name)

    # Build allowed file terms from active topics only
    allowed_file_terms = set()
    for topic_name in active_topics:
        allowed_file_terms.update(topic_groups[topic_name]["file_terms"])

    # Also add the original meaningful terms (for direct matches)
    allowed_file_terms.update(meaningful_terms)

    retrieval_logger.debug(f"  Query terms: {sorted(meaningful_terms)}")
    retrieval_logger.debug(f"  Active topics: {sorted(active_topics)}")
    retrieval_logger.debug(f"  Allowed file terms: {sorted(allowed_file_terms)}")

    # Collect all matches with scores
    all_matches = []  # (score, path)
    seen_files = set()  # avoid duplicates from overlapping search dirs

    # Relevant directories to search
    search_dirs = [
        os.path.join("data", "forms"),
        os.path.join("data", "policies"),
        os.path.join("data", "manuals"),
        "data"
    ]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        for root, _, files in os.walk(search_dir):
            for file in files:
                if not file.lower().endswith(".pdf"):
                    continue

                full_path = os.path.join(root, file)
                if full_path in seen_files:
                    continue
                seen_files.add(full_path)

                file_name_clean = file.lower().replace(".pdf", "").replace("_", " ").replace("-", " ").replace(".", " ")

                # Extract meaningful words from filename (skip filename stop words)
                file_words = {w.strip() for w in file_name_clean.split() if w.strip() and len(w.strip()) > 1}
                file_words -= filename_stop_words

                score = 0

                # Score: only count matches against allowed_file_terms (topic-filtered)
                for term in allowed_file_terms:
                    if len(term) < 3:
                        continue
                    if term in file_name_clean:
                        # Longer terms = more specific = higher weight
                        weight = 3 if len(term) >= 6 else 2 if len(term) >= 4 else 1
                        score += weight

                # Penalize files that have prominent terms NOT in the query's topic
                # This prevents cross-topic pollution
                if active_topics and score > 0:
                    for other_topic, other_info in topic_groups.items():
                        if other_topic not in active_topics:
                            # Check if this file's name strongly indicates another topic
                            other_file_terms = other_info["file_terms"]
                            cross_matches = file_words & other_file_terms
                            if len(cross_matches) >= 2:
                                # File belongs to a different topic — heavy penalty
                                score -= 4 * len(cross_matches)
                                retrieval_logger.debug(
                                    f"  Cross-topic penalty for {file}: "
                                    f"topic '{other_topic}' terms {cross_matches}"
                                )

                if score >= 3:  # Require stronger matches (was 2)
                    all_matches.append((score, full_path))

    # Sort by score (highest first) and return top results
    all_matches.sort(key=lambda x: -x[0])
    results = [path for _, path in all_matches[:max_results]]

    if results:
        for score, path in all_matches[:max_results]:
            retrieval_logger.info(f"  File match: {os.path.basename(path)} (score={score})")
    else:
        retrieval_logger.debug("No matching files found")

    return results


def save_chat_history(history: List, session_id: str):
    """
    Saves the current chat history to a JSON file.
    
    Args:
        history: Conversation history
        session_id: Unique session identifier
    """
    filepath = os.path.join(str(LOG_DIR), f"chat_{session_id}.json")
    
    data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "history": history
    }
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.debug(f"Chat history saved: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")


def convert_gradio_history_to_langchain(history: List) -> List:
    """
    Convert Gradio message history to LangChain message format.
    
    Args:
        history: List of dicts with 'role' and 'content' keys
    
    Returns:
        List of HumanMessage and AIMessage objects
    """
    api_history = []
    
    for msg_data in history:
        role = msg_data.get('role')
        content = msg_data.get('content', '')
        
        # Handle complex content
        if isinstance(content, dict):
            content = content.get('text', str(content))
        elif isinstance(content, list):
            content = str(content)
        
        if role == 'user':
            api_history.append(HumanMessage(content=str(content)))
        elif role == 'assistant':
            api_history.append(AIMessage(content=str(content)))
    
    return api_history


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Initialize on import
if not initialize_system():
    logger.error("System initialization failed! The assistant may not work correctly.")
