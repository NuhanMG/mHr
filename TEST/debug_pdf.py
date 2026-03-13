from langchain_community.document_loaders import PyPDFLoader
import os

file_path = "data/forms/Salary Advance Form.pdf"

print(f"Checking file: {file_path}")
if not os.path.exists(file_path):
    print("File does not exist!")
else:
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("No documents loaded (empty list).")
        else:
            print(f"Loaded {len(docs)} pages.")
            for i, doc in enumerate(docs):
                content = doc.page_content.strip()
                print(f"--- Page {i+1} Content Start ---")
                print(content if content else "[No text content found - likely scanned image]")
                print(f"--- Page {i+1} Content End ---")
                print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error loading PDF: {e}")
