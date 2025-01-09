__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from agentic_rag import initialize_app, update_llm
import sys
import io
import os

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Smart Business Guide",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar layout
with st.sidebar:
    try:
        st.image("LOGO_UPBEAT.jpg", width=150, use_container_width=True)
    except Exception as e:
        st.warning("Unable to load image. Continuing without it.")

    st.title("📘 Smart Guide 1.0")
    st.markdown("**Actions:**")
    model_list = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",  
    "llama-3.1-8b-instant", 
    "llama3-8b-8192", 
    "mixtral-8x7b-32768", 
    "gemma2-9b-it"
    # "gpt-4o-mini",
    # "gpt-4o"
    ]
    # Modify the sidebar to include model selection
    model = st.sidebar.selectbox("Select Model", model_list)

    search_option = st.sidebar.radio(
    "Search options",
    ["Smart guide + tools", "Internet search only", "Hybrid search (Guides + internet)"],
    index=0
    )

    # Set the corresponding boolean values based on the selected option
    hybrid_search = search_option == "Hybrid search (Guides + internet)"
    internet_search = search_option == "Internet search only"
    
    reset_button = st.button("Reset Conversation")

    # Initialize the app with the selected model
    app = initialize_app(model, hybrid_search, internet_search)
    if reset_button:
        st.session_state.messages = []
# Title
st.title("📘 Smart Guide for Entrepreneurship and Business Planning in Finland")
st.markdown(
    """
    <div style="text-align: left; font-size: 18px; margin-top: 20px; line-height: 1.6;">
        🤖 <b>Welcome to your Smart Business Guide!</b><br>
        I am here to assist you with:<br>
        <ul style="list-style-position: inside; text-align: left; display: inline-block;">
            <li>Finding answers from business and entrepreneurship guides in Finland</li>
            <li>Providing up-to-date information through internet search</li>
            <li>Specialized guidance for tax-related information, permits & licenses, business registration, residence permits, etc. :</li>
        </ul>
        <p style="margin-top: 10px;"><b>Start by typing your question in the chat below, and I'll provide tailored answers for your business needs!</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f"**Assistant:** {message['content']}")

# Input box at the bottom for new messages
if user_input := st.chat_input("Type your question (Max. 100 char):"):
    if len(user_input) > 100:
        st.error("Your question exceeds 100 characters. Please shorten it and try again.")
    else:
        # Add user's message to session state and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_input}")

        # Capture print statements from agentic_rag.py
        output_buffer = io.StringIO()
        sys.stdout = output_buffer  # Redirect stdout to the buffer

        try:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                debug_placeholder = st.empty()
                streamed_response = ""

                # Show spinner while streaming the response
                with st.spinner("Thinking..."):
                    #inputs = {"question": user_input}
                    inputs = {"question": user_input, "hybrid_search": hybrid_search, "internet_search":internet_search}
                    for i, output in enumerate(app.stream(inputs)):
                        # Capture intermediate print messages
                        debug_logs = output_buffer.getvalue()
                        debug_placeholder.text_area(
                            "Debug Logs",
                            debug_logs,
                            height=100,
                            key=f"debug_logs_{i}"
                        )

                        if "generate" in output and "generation" in output["generate"]:
                            # Append new content to streamed response
                            streamed_response += output["generate"]["generation"]
                            # Update the placeholder with the streamed response so far
                            response_placeholder.markdown(f"**Assistant:** {streamed_response}")

                # Store the final response in session state
                st.session_state.messages.append({"role": "assistant", "content": streamed_response or "No response generated."})
        except Exception as e:
            # Handle errors and display in the conversation history
            error_message = f"An error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)
        finally:
            # Restore stdout to its original state
            sys.stdout = sys.__stdout__
