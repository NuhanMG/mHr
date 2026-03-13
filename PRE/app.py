import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import json
import uuid
from datetime import datetime

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore"
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
LOG_DIR = "chat_logs"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# --- Global Components ---
vectorstore = None
rag_chain = None

def initialize_system():
    global vectorstore, rag_chain
    
    if os.path.exists(VECTORSTORE_PATH):
        print(f"Loading vectorstore from {VECTORSTORE_PATH}...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚠️ Vectorstore not found. Please run ingest.py first.")
        return False

    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # --- 1. Contextualize Question Chain ---
    # This chain rewrites the user's question based on chat history to make it standalone.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- 2. QA Chain ---
    # This chain answers the question using the retrieved context.
    qa_system_prompt = (
        "You are a helpful HR Assistant for Mobitel Sri Lanka. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the information is not in the context, say 'Sorry, the requested information is not available in the HR documents.' "
        "and do not attempt to fabricate an answer. "
        "Keep the answer professional, concise, and helpful. "
        "\n\n"
        "IMPORTANT: The context below includes the 'Source' for each piece of information. "
        "If the user specifically asks for a form, policy, or document, "
        "you MUST explicitly mention the ACTUAL Source filename from the context at the end of your response in this exact format: "
        "[[FILE:path/to/file.pdf]] "
        "Use the exact path provided in the 'Source' field. Do not invent paths."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Define how documents are formatted in the context (include source)
    document_prompt = ChatPromptTemplate.from_template("Content: {page_content}\nSource: {source}")
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return True

# Initialize on startup
initialize_system()

def find_best_matching_file(query):
    """
    Searches for the best matching file in the data directory based on query keywords.
    """
    query_lower = query.lower()
    best_match = None
    max_matches = 0
    
    # relevant directories
    search_dirs = [os.path.join("data", "forms"), os.path.join("data", "policies"), "data"]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, _, files in os.walk(search_dir):
            for file in files:
                if not file.lower().endswith(".pdf"):
                    continue
                
                # Simple keyword matching
                file_keywords = file.lower().replace(".pdf", "").split()
                matches = sum(1 for word in query_lower.split() if word in file.lower())
                
                # Bonus for exact substring match
                if file.lower().replace(".pdf","") in query_lower:
                    matches += 5
                
                if matches > max_matches:
                    max_matches = matches
                    best_match = os.path.join(root, file)
    
    # Threshold for considering it a match (at least 2 keywords or 1 strong keyword)
    if max_matches >= 2:
        return best_match
    return None

def get_answer(message, history):
    """
    Get Answer from RAG Chain.
    history: List of LangChain messages (HumanMessage, AIMessage)
    """
    # Defensive programming: ensure message is string
    if not isinstance(message, str):
         message = str(message)
        
    if rag_chain is None:
        return "⚠️ System not initialized based on data. Please contact admin.", None

    file_path = None
    answer = ""

    try:
        response = rag_chain.invoke({"input": message, "chat_history": history})
        answer = response["answer"]
        print(f"DEBUG: LLM Response: {answer}")  # Debug print
        
        # Check for file download intent in the answer
        if "[[FILE:" in answer:
            # Extract the file path
            start = answer.find("[[FILE:") + 7
            end = answer.find("]]", start)
            if end != -1:
                extracted_path = answer[start:end].strip()
                print(f"DEBUG: Extracted path from LLM: {extracted_path}")
                
                if os.path.exists(extracted_path):
                     file_path = extracted_path
                else:
                     # Try absolute
                     abs_path = os.path.abspath(extracted_path)
                     if os.path.exists(abs_path):
                         file_path = abs_path

                # Cleanup answer
                answer = answer.replace(f"[[FILE:{extracted_path}]]", "").strip()

        # FALLBACK: If no file found from LLM, but user asked for a form/application
        if not file_path and ("form" in message.lower() or "application" in message.lower()):
            print("DEBUG: User asked for form, but LLM didn't provide valid path. Searching manually...")
            fallback_file = find_best_matching_file(message)
            if fallback_file:
                print(f"DEBUG: Found fallback file: {fallback_file}")
                file_path = fallback_file
                if "[[FILE:" not in answer: # Don't double append if LLM tried
                     answer += f"\n\nI found a relevant form for you: {os.path.basename(file_path)}"

    except Exception as e:
        answer = f"⚠️ An error occurred: {str(e)}"
        print(f"ERROR: {e}")
        file_path = None

    return answer, file_path

# --- Gradio UI ---

def save_chat_history(history, session_id):
    """
    Saves the current chat history to a JSON file.
    """
    filepath = os.path.join(LOG_DIR, f"chat_{session_id}.json")
    
    data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "history": history
    }
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"ERROR saving chat history: {e}")

def bot_turn(history, session_id):
    # history is list of dicts.
    if not history:
        return history, None
        
    last_message_data = history[-1]
    # In 'messages' type, content can be complex.
    user_message_content = last_message_data.get('content', '')
    
    # Parse history for context (everything before the last message)
    api_history = []
    for msg_data in history[:-1]:
        role = msg_data.get('role')
        content = msg_data.get('content', '')
        
        # Handle complex content in history too if needed
        if isinstance(content, dict):
            content = content.get('text', str(content))
        elif isinstance(content, list):
            content = str(content)
            
        if role == 'user':
            api_history.append(HumanMessage(content=str(content)))
        elif role == 'assistant':
            api_history.append(AIMessage(content=str(content)))

    # Get response
    answer, file_path = get_answer(str(user_message_content), api_history)
    
    # Append assistant answer
    history.append({"role": "assistant", "content": answer})
    
    # Save history
    save_chat_history(history, session_id)
    
    return history, file_path

with gr.Blocks(theme=gr.themes.Soft(), title="Mobitel HR Assistant") as demo:
    # Session state to track unique users. Initialize with a UUID.
    session_id_state = gr.State(lambda: str(uuid.uuid4()))

    gr.Markdown(
        """
        # 🤖 Mobitel HR Assistant
        Welcome! Ask me anything about HR policies, leave procedures, or request forms.
        """
    )
    
    with gr.Row():
        chatbot = gr.Chatbot(height=500, label="Conversation", type="messages")
        
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your question here...", container=False, scale=4)
        submit_btn = gr.Button("Send", scale=1)
        
    file_output = gr.File(label="Download Requested Document")

    def user_turn(user_message, history):
        # Format for type="messages": List of dicts [{"role": "user", "content": "msg"}, ...]
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, session_id_state], [chatbot, file_output]
    )
    
    submit_btn.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, session_id_state], [chatbot, file_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
