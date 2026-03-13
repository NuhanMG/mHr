"""
Mobitel HR Assistant - Utility Functions
Contains validation, query processing, caching, and helper functions.
"""

import re
import time
import hashlib
from collections import OrderedDict
from typing import Optional, Tuple, List
from functools import wraps

from config import (
    setup_logging,
    MIN_QUERY_LENGTH,
    MAX_QUERY_LENGTH,
    QUERY_EXPANSION_MAP,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)

# Setup logger for utils
logger = setup_logging("hr_assistant.utils")


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_input(message: str) -> Tuple[bool, str, str]:
    """
    Validate and sanitize user input.
    
    Args:
        message: Raw user input string
        
    Returns:
        Tuple of (is_valid, sanitized_message, error_message)
    """
    logger.debug(f"Validating input: '{message[:50]}...' (length: {len(message) if message else 0})")
    
    # Check for None or empty
    if message is None:
        logger.warning("Input validation failed: None input")
        return False, "", "Please enter a question."
    
    # Convert to string if needed
    if not isinstance(message, str):
        message = str(message)
    
    # Strip whitespace
    message = message.strip()
    
    # Check minimum length
    if len(message) < MIN_QUERY_LENGTH:
        logger.warning(f"Input validation failed: Too short ({len(message)} chars)")
        return False, "", f"Please provide a more detailed question (at least {MIN_QUERY_LENGTH} characters)."
    
    # Check maximum length
    if len(message) > MAX_QUERY_LENGTH:
        logger.warning(f"Input validation failed: Too long ({len(message)} chars)")
        return False, "", f"Your question is too long. Please limit to {MAX_QUERY_LENGTH} characters."
    
    # Sanitize - remove potential prompt injection patterns
    sanitized = sanitize_input(message)
    
    logger.info(f"Input validation passed: '{sanitized[:50]}...'")
    return True, sanitized, ""


def sanitize_input(message: str) -> str:
    """
    Sanitize input to prevent prompt injection attacks.
    
    Args:
        message: User input string
        
    Returns:
        Sanitized string
    """
    # Remove common prompt injection patterns
    injection_patterns = [
        r"ignore\s+(previous|above|all)\s+instructions?",
        r"disregard\s+(previous|above|all)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(if\s+you\s+are|a)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"\[\s*INST\s*\]",
    ]
    
    sanitized = message
    for pattern in injection_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(f"Potential prompt injection detected: pattern '{pattern}'")
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()


# ============================================================================
# QUERY EXPANSION
# ============================================================================

def expand_query(query: str) -> str:
    """
    Expand query with HR domain synonyms for better retrieval.
    
    Non-tech users often use informal terms. This function adds
    formal HR terminology to improve search results.
    
    Args:
        query: Original user query
        
    Returns:
        Expanded query with HR terms appended
    """
    logger.debug(f"Expanding query: '{query}'")
    
    query_lower = query.lower()
    expansions = []
    
    for informal, formal in QUERY_EXPANSION_MAP.items():
        if informal.lower() in query_lower and formal.lower() not in query_lower:
            expansions.append(formal)
            logger.debug(f"  Added expansion: '{informal}' -> '{formal}'")
    
    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
        logger.info(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded
    
    logger.debug("No expansions applied")
    return query


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter for session-based throttling.
    """
    
    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, 
                 window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # session_id -> list of timestamps
        self.logger = setup_logging("hr_assistant.rate_limiter")
    
    def is_allowed(self, session_id: str) -> Tuple[bool, str]:
        """
        Check if a request is allowed for the given session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Tuple of (is_allowed, message)
        """
        current_time = time.time()
        
        # Initialize if new session
        if session_id not in self.requests:
            self.requests[session_id] = []
        
        # Clean old requests outside the window
        self.requests[session_id] = [
            t for t in self.requests[session_id] 
            if current_time - t < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[session_id]) >= self.max_requests:
            wait_time = int(self.window_seconds - (current_time - self.requests[session_id][0]))
            self.logger.warning(f"Rate limit exceeded for session {session_id[:8]}...")
            return False, f"You're sending too many requests. Please wait {wait_time} seconds."
        
        # Record this request
        self.requests[session_id].append(current_time)
        self.logger.debug(f"Rate limit check passed for session {session_id[:8]}... ({len(self.requests[session_id])}/{self.max_requests})")
        return True, ""
    
    def cleanup_old_sessions(self, max_age_seconds: int = 3600):
        """Remove sessions that haven't been active for a while."""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, timestamps in self.requests.items():
            if not timestamps or current_time - max(timestamps) > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.requests[session_id]
        
        if sessions_to_remove:
            self.logger.debug(f"Cleaned up {len(sessions_to_remove)} old sessions")


# ============================================================================
# CACHING
# ============================================================================

class QueryCache:
    """
    Simple LRU cache with TTL for query results.
    """
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()  # key -> (value, timestamp)
        self.logger = setup_logging("hr_assistant.cache")
    
    def _generate_key(self, query: str, history_hash: str = "") -> str:
        """Generate a cache key from query and history."""
        content = f"{query}|{history_hash}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, history_hash: str = "") -> Optional[Tuple[str, Optional[str]]]:
        """
        Get cached result if available and not expired.
        
        Args:
            query: User query
            history_hash: Hash of conversation history for context
            
        Returns:
            Cached (answer, file_path) tuple or None
        """
        key = self._generate_key(query, history_hash)
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.logger.debug(f"Cache miss (expired): '{query[:30]}...'")
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.logger.info(f"Cache hit: '{query[:30]}...'")
            return value
        
        self.logger.debug(f"Cache miss: '{query[:30]}...'")
        return None
    
    def set(self, query: str, result: Tuple[str, Optional[str]], history_hash: str = ""):
        """
        Store result in cache.
        
        Args:
            query: User query
            result: (answer, file_path) tuple
            history_hash: Hash of conversation history
        """
        key = self._generate_key(query, history_hash)
        
        # Remove oldest if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.logger.debug("Cache eviction (LRU)")
        
        self.cache[key] = (result, time.time())
        self.logger.debug(f"Cache set: '{query[:30]}...'")
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.logger.info("Cache cleared")


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_confidence(query: str, answer: str, source_docs: List) -> Tuple[float, str]:
    """
    Calculate confidence score for the generated answer.
    
    Args:
        query: Original user query
        answer: Generated answer
        source_docs: List of source documents used
        
    Returns:
        Tuple of (confidence_score, confidence_level)
    """
    logger.debug(f"Calculating confidence for query: '{query[:30]}...'")
    
    score = 0.0
    factors = []
    
    # Factor 1: Number of source documents
    num_docs = len(source_docs) if source_docs else 0
    if num_docs >= 3:
        score += 0.3
        factors.append(f"Good document coverage ({num_docs} docs)")
    elif num_docs >= 1:
        score += 0.15
        factors.append(f"Limited document coverage ({num_docs} docs)")
    else:
        factors.append("No source documents")
    
    # Factor 2: Answer length (too short or too long may indicate issues)
    answer_len = len(answer)
    if 50 <= answer_len <= 1000:
        score += 0.2
        factors.append("Appropriate answer length")
    elif answer_len < 50:
        score += 0.05
        factors.append("Answer may be too brief")
    else:
        score += 0.1
        factors.append("Answer may be too verbose")
    
    # Factor 3: Check for uncertainty phrases
    uncertainty_phrases = [
        "i'm not sure", "i don't know", "not available",
        "cannot find", "no information", "unclear"
    ]
    answer_lower = answer.lower()
    has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
    if not has_uncertainty:
        score += 0.25
        factors.append("No uncertainty phrases")
    else:
        factors.append("Contains uncertainty phrases")
    
    # Factor 4: Query terms present in answer
    query_terms = set(query.lower().split())
    answer_terms = set(answer_lower.split())
    overlap = len(query_terms & answer_terms) / max(len(query_terms), 1)
    if overlap >= 0.3:
        score += 0.25
        factors.append(f"Good query-answer overlap ({overlap:.0%})")
    else:
        score += overlap * 0.25
        factors.append(f"Limited query-answer overlap ({overlap:.0%})")
    
    # Determine confidence level
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        level = "HIGH"
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    logger.info(f"Confidence: {score:.2f} ({level}) - Factors: {factors}")
    return score, level


def format_low_confidence_answer(answer: str, level: str) -> str:
    """
    Add a disclaimer to low-confidence answers.
    
    Args:
        answer: Original answer
        level: Confidence level (HIGH, MEDIUM, LOW)
        
    Returns:
        Formatted answer with disclaimer if needed
    """
    if level == "LOW":
        return f"⚠️ I found some information but I'm not fully confident in this answer:\n\n{answer}\n\n*Please verify this information with HR if it's critical.*"
    return answer


# ============================================================================
# DOCUMENT HELPERS
# ============================================================================

def get_document_category(file_path: str) -> str:
    """
    Determine document category from file path.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Category string (form, policy, manual, faq, holiday, other)
    """
    path_lower = file_path.lower()
    
    if "forms" in path_lower or "form" in path_lower:
        return "form"
    elif "policies" in path_lower or "policy" in path_lower:
        return "policy"
    elif "manuals" in path_lower or "manual" in path_lower:
        return "manual"
    elif "faq" in path_lower:
        return "faq"
    elif "holiday" in path_lower:
        return "holiday"
    else:
        return "other"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract key terms from document text.
    
    Args:
        text: Document text content
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    # Common HR keywords to look for
    hr_keywords = [
        "leave", "annual", "medical", "salary", "advance", "policy",
        "form", "application", "travel", "training", "discipline",
        "harassment", "resignation", "transfer", "phone", "identity",
        "card", "bonus", "increment", "holiday", "visa", "allowance"
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in hr_keywords:
        if keyword in text_lower and keyword not in found_keywords:
            found_keywords.append(keyword)
            if len(found_keywords) >= max_keywords:
                break
    
    return found_keywords


# Global instances
rate_limiter = RateLimiter()
query_cache = QueryCache()
