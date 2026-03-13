"""
Mobitel HR Assistant - Document Ingestion Script
This script loads documents, applies intelligent chunking, enriches metadata,
and creates a FAISS vectorstore for the RAG system.

Features:
- Document-type specific chunking strategies
- Rich metadata enrichment (category, type, keywords)
- Comprehensive logging
- Error handling and validation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    setup_logging,
    DATA_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHUNKING_CONFIG,
    DOCUMENT_CATEGORIES,
)
from utils import get_document_category, extract_keywords

# Setup logger
logger = setup_logging("hr_assistant.ingest")


def get_chunking_config(file_path: str) -> dict:
    """
    Get document-type specific chunking configuration.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dictionary with chunk_size and chunk_overlap
    """
    category = get_document_category(file_path)
    config = CHUNKING_CONFIG.get(category, CHUNKING_CONFIG["default"])
    logger.debug(f"  Chunking config for {category}: {config}")
    return config


def enrich_metadata(doc, file_path: str) -> None:
    """
    Enrich document metadata with additional information.
    
    Args:
        doc: LangChain document object
        file_path: Original file path
    """
    # Get category from folder structure
    category = get_document_category(file_path)
    doc.metadata['category'] = category
    
    # Determine document type
    file_name = os.path.basename(file_path).lower()
    if 'form' in file_name or 'application' in file_name:
        doc.metadata['doc_type'] = 'form'
    elif 'policy' in file_name:
        doc.metadata['doc_type'] = 'policy'
    elif 'manual' in file_name or 'guide' in file_name:
        doc.metadata['doc_type'] = 'manual'
    elif 'faq' in file_name or category == 'faq':
        doc.metadata['doc_type'] = 'faq'
    else:
        doc.metadata['doc_type'] = 'document'
    
    # Extract keywords from content
    keywords = extract_keywords(doc.page_content)
    doc.metadata['keywords'] = ', '.join(keywords)
    
    # Keep original source path for file downloads
    doc.metadata['source'] = file_path


def create_splitter_for_document(file_path: str) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with document-specific configuration.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    config = get_chunking_config(file_path)
    category = get_document_category(file_path)
    
    # Customize separators based on document type
    if category == 'faq':
        separators = ["\n\n", "Q:", "A:", "\n", " "]
    elif category == 'policy':
        separators = ["\n\n\n", "\n\n", ".\n", "\n", ". ", " "]
    else:
        separators = ["\n\n", "\n", " ", ""]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        separators=separators,
        length_function=len,
    )


def ingest_documents():
    """
    Main ingestion function.
    Loads PDFs, applies intelligent chunking, enriches metadata,
    and saves a FAISS vectorstore for the RAG system.
    """
    logger.info("="*60)
    logger.info("STARTING DOCUMENT INGESTION")
    logger.info("="*60)
    
    # Validation
    if not os.path.exists(str(DATA_PATH)):
        logger.error(f"Data directory not found: {DATA_PATH}")
        return False
    
    # Chunk collection
    all_chunks = []
    files_processed = 0
    files_failed = 0
    category_counts = {}
    
    # Walk through the data directory
    logger.info(f"Scanning directory: {DATA_PATH}")
    for root, dirs, files in os.walk(str(DATA_PATH)):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
                
            file_path = os.path.join(root, file)
            logger.info(f"\nProcessing: {file}")
            
            try:
                # Step 1: Load document
                logger.debug(f"  Step 1: Loading PDF...")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                logger.debug(f"  → Loaded {len(docs)} pages")
                
                # Step 2: Enrich metadata for each page
                logger.debug(f"  Step 2: Enriching metadata...")
                for doc in docs:
                    enrich_metadata(doc, file_path)
                
                category = docs[0].metadata.get('category', 'unknown') if docs else 'unknown'
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Step 3: Create document-specific splitter
                logger.debug(f"  Step 3: Creating splitter for category '{category}'...")
                splitter = create_splitter_for_document(file_path)
                
                # Step 4: Split documents
                logger.debug(f"  Step 4: Splitting into chunks...")
                chunks = splitter.split_documents(docs)
                logger.info(f"  → Created {len(chunks)} chunks (category: {category})")
                
                all_chunks.extend(chunks)
                
                files_processed += 1
                
            except Exception as e:
                logger.error(f"  ✗ Error processing {file}: {e}")
                files_failed += 1
    
    logger.info("\n" + "-"*60)
    logger.info("DOCUMENT LOADING SUMMARY")
    logger.info("-"*60)
    logger.info(f"Files processed: {files_processed}")
    logger.info(f"Files failed: {files_failed}")
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Categories: {category_counts}")
    
    if not all_chunks:
        logger.error("No chunks created! Check if PDFs exist in the data directory.")
        return False
    
    # Initialize embeddings model once
    logger.info(f"\nInitializing embeddings model: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # ---- Create Vectorstore (ALL documents) ----
    logger.info("\n" + "-"*60)
    logger.info("CREATING VECTORSTORE")
    logger.info("-"*60)
    
    
    try:
        logger.info(f"Creating FAISS index with {len(all_chunks)} chunks...")
        logger.info("(This may take a few minutes...)")
        
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(str(VECTORSTORE_PATH))
        logger.info(f"✓ Vectorstore saved to: {VECTORSTORE_PATH}")
        
    except Exception as e:
        logger.error(f"✗ Failed to create general vectorstore: {e}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("INGESTION COMPLETE!")
    logger.info("="*60)
    
    # Auto-parse Holiday PDF if present
    logger.info("\n" + "-"*60)
    logger.info("POST-INGEST: Holiday Parser")
    logger.info("-"*60)
    try:
        from holiday_parser import parse_holiday_pdf
        holiday_success = parse_holiday_pdf()
        if holiday_success:
            logger.info("✓ Holiday data auto-extracted to holidays.json")
        else:
            logger.warning("⚠ Holiday parser did not find a Holiday PDF or extraction failed")
    except Exception as e:
        logger.warning(f"⚠ Holiday parser skipped: {e}")
    
    # Print summary
    logger.info(f"\nVectorstore created:")
    logger.info(f"  📁 {VECTORSTORE_PATH} ({len(all_chunks)} chunks)")
    
    # Print chunk samples for verification
    logger.info("\nSample chunks (first 2):")
    for i, chunk in enumerate(all_chunks[:2]):
        logger.info(f"\n--- Chunk {i+1} ---")
        logger.info(f"Category: {chunk.metadata.get('category', 'N/A')}")
        logger.info(f"Source: {chunk.metadata.get('source', 'N/A')}")
        logger.info(f"Content preview: {chunk.page_content[:150]}...")
    
    return True


if __name__ == "__main__":
    success = ingest_documents()
    if not success:
        logger.error("Ingestion failed!")
        sys.exit(1)
