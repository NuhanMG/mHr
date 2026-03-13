"""
Mobitel HR Assistant - Frontend (Gradio UI)
Run this script to start the HR Assistant chatbot.

Features:
- Modern chat interface with FAQ accordion panel
- Browsable FAQ sections for quick answers
- Example questions for non-tech users
- File download for forms
- Session tracking
"""

import gradio as gr
import uuid

# Import backend functions
from backend import (
    get_answer,
    save_chat_history,
    convert_gradio_history_to_langchain
)
from config import setup_logging
from faq_data import get_all_faqs

# Setup logger
logger = setup_logging("hr_assistant.frontend")


def bot_turn(history, session_id):
    """
    Process the bot's response to the user's message.
    
    Args:
        history: Conversation history
        session_id: Unique session identifier
        
    Returns:
        Updated history, file paths list, file visibility
    """
    if not history:
        return history, [], gr.update(visible=False)
    
    last_message_data = history[-1]
    user_message_content = last_message_data.get('content', '')
    
    logger.info(f"Processing user message: '{str(user_message_content)[:50]}...'")
    
    # Parse history for context (everything before the last message)
    api_history = convert_gradio_history_to_langchain(history[:-1])

    # Get response from RAG backend (returns answer + list of file paths)
    answer, file_paths = get_answer(str(user_message_content), api_history, session_id)
    
    # Append assistant answer
    history.append({"role": "assistant", "content": answer})
    
    # Save history
    save_chat_history(history, session_id)
    
    # Show file download only if file paths were returned
    file_visible = len(file_paths) > 0
    
    logger.info(f"Response generated. Files to download: {len(file_paths)}")
    
    return history, file_paths if file_visible else [], gr.update(visible=file_visible)


def user_turn(user_message, history):
    """
    Process the user's message submission.
    
    Args:
        user_message: User's input text
        history: Current conversation history
        
    Returns:
        Empty textbox, updated history
    """
    if history is None:
        history = []
    
    if not user_message or not user_message.strip():
        return "", history
    
    return "", history + [{"role": "user", "content": user_message}]


def faq_click(question, history, session_id):
    """
    Handle FAQ question click — send it as a user message and get response.
    
    Args:
        question: The FAQ question text
        history: Current conversation history
        session_id: Unique session identifier
        
    Returns:
        Empty textbox, updated history with Q&A, file path, file visibility
    """
    if history is None:
        history = []
    
    # Add user question to history
    history.append({"role": "user", "content": question})
    
    # Get answer from backend
    api_history = convert_gradio_history_to_langchain(history[:-1])
    answer, file_path = get_answer(question, api_history, session_id)
    
    # Add bot response
    history.append({"role": "assistant", "content": answer})
    save_chat_history(history, session_id)
    
    file_visible = file_path is not None
    return "", history, file_path, gr.update(visible=file_visible)


# --- Build FAQ accordion HTML ---
def build_faq_accordions():
    """Build Gradio accordion components for each FAQ category."""
    faq_data = get_all_faqs()
    components = []
    for category, faqs in faq_data.items():
        components.append((category, faqs))
    return components


# --- Gradio UI ---
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Mobitel HR Assistant",
    css="""
        .gradio-container { max-width: 950px !important; }
        .faq-btn { 
            text-align: left !important; 
            justify-content: flex-start !important;
            font-size: 13px !important;
            padding: 8px 12px !important;
            margin: 2px 0 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            background: #f8f9fa !important;
            color: #333 !important;
            cursor: pointer !important;
        }
        .faq-btn:hover {
            background: #e3f2fd !important;
            border-color: #90caf9 !important;
        }
    """
) as demo:
    
    # Session state
    session_id_state = gr.State(lambda: str(uuid.uuid4()))

    # Header
    gr.Markdown(
        """
        # 🤖 Mobitel HR Assistant
        
        Welcome! I can help you with HR policies, leave procedures, forms, and more.
        Ask your questions in English
        """
    )
    
    with gr.Tabs():
        # --- TAB 1: Chat ---
        with gr.TabItem("💬 Chat", id="chat_tab"):
            # Example questions
            gr.Markdown("### 💡 Try asking:")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    - "What is the annual leave policy?"
                    - "How do I apply for salary advance?"
                    - "I want to take a vacation next month"
                    """)
                with gr.Column(scale=1):
                    gr.Markdown("""
                    - "Download the visa application form"
                    - "Suggest me good days for 4 day leave"
                    - "How to report harassment?"
                    """)
            
            gr.Markdown("---")
            
            # Chat interface
            chatbot = gr.Chatbot(
                height=420,
                label="Conversation",
                type="messages",
                show_label=False,
                avatar_images=(None, "🤖")
            )
            
            # Input area
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your question here... (e.g., 'How do I apply for leave?')",
                    container=False,
                    scale=4,
                    autofocus=True
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")
            
            # Download section - initially hidden
            with gr.Row():
                file_output = gr.File(
                    label="📄 Download Documents",
                    file_count="multiple",
                    visible=False
                )
        
        # --- TAB 2: FAQ Browser ---
        with gr.TabItem("📋 FAQs", id="faq_tab"):
            gr.Markdown(
                """
                ### Browse Frequently Asked Questions
                Click any question to get the answer in the chat.
                """
            )
            
            faq_data = get_all_faqs()
            faq_buttons = []
            
            for category, faqs in faq_data.items():
                with gr.Accordion(category, open=False):
                    for question, answer in faqs:
                        btn = gr.Button(
                            f"❓ {question}",
                            elem_classes=["faq-btn"],
                            size="sm"
                        )
                        faq_buttons.append((btn, question))

    # --- Event handlers ---
    
    # Chat submit handlers
    msg.submit(
        user_turn, 
        [msg, chatbot], 
        [msg, chatbot], 
        queue=False
    ).then(
        bot_turn, 
        [chatbot, session_id_state], 
        [chatbot, file_output, file_output]
    )
    
    submit_btn.click(
        user_turn, 
        [msg, chatbot], 
        [msg, chatbot], 
        queue=False
    ).then(
        bot_turn, 
        [chatbot, session_id_state], 
        [chatbot, file_output, file_output]
    )
    
    # FAQ button click handlers — each button sends its question to chat
    for btn, question in faq_buttons:
        # Use a wrapper that returns the question as if the user typed it
        btn.click(
            fn=lambda q=question: ("", [{"role": "user", "content": q}]),
            inputs=None,
            outputs=[msg, chatbot],
            queue=False
        ).then(
            bot_turn,
            [chatbot, session_id_state],
            [chatbot, file_output, file_output]
        )


if __name__ == "__main__":
    logger.info("Starting Mobitel HR Assistant...")
    demo.launch(inbrowser=True)
