from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "nomic-embed-text"

print(f"Loading vectorstore from {VECTORSTORE_PATH}...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
try:
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    query = "salary advance form"
    print(f"\nSearching for: '{query}'")
    results = vectorstore.similarity_search(query, k=5)
    
    if not results:
        print("No results found.")
    else:
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content Preview: {doc.page_content[:200]}...")
except Exception as e:
    print(f"Error: {e}")
