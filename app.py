import io
import re  # ensure regex is imported
import sys
import time  # for minimal delay

import streamlit as st
import torch
import tornado
from langchain_openai import ChatOpenAI

# Early session state initialization
default_keys = {
    "messages": [],
    "followup_key": 0,
    "pending_followup": None,
    "last_assistant": None,
    "followup_questions": [],
    "selected_model": "gpt-4o",
    "selected_routing_model": "gpt-4o",
    "selected_grading_model": "gpt-4o",
    "selected_embedding_model": "text-embedding-3-large",
    "hybrid_search": False,
    "internet_search": False,
    "answer_style": "Explanatory",
    # Include any additional keys used later (e.g., llm, embed_model)
}
for key, default in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

time.sleep(0.1)  # optional delay to ensure proper initialization

from agentic_rag import initialize_app
from st_callback import get_streamlit_cb

# Fix below for "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!"
torch.classes.__path__ = []

# TODO: App crashes when selecting "llama-3.1-8b-instant" model.
st.set_option("client.showErrorDetails", False)  # Hide error details

# -------------------- Initialization --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "followup_key" not in st.session_state:
    st.session_state.followup_key = 0
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None
if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = None
if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = []

# -------------------- Helper Functions --------------------


def get_followup_questions(last_user, last_assistant):
    """
    Generate three concise follow-up questions dynamically based on the latest conversation.
    """
    prompt = f"""Based on the conversation below:
User: {last_user}
Assistant: {last_assistant}
Generate three concise follow-up questions that a user might ask next.
Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question. Focus on brevity.
Follow-up Questions:"""
    try:
        # Use ChatOpenAI as a fallback if the selected models because otherwise it will fail. e.g Gemma might not support invoking method.
        if any(model_type in st.session_state.selected_model.lower() 
               for model_type in ["gemma2", "deepseek", "mixtral"]):
            fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
            response = fallback_llm.invoke(prompt)
        else:
            response = st.session_state.llm.invoke(prompt)

        text = response.content if hasattr(
            response, "content") else str(response)
        questions = [q.strip() for q in text.split('\n') if q.strip()]
        return questions[:3]
    except Exception as e:
        print(f"Failed to generate follow-up questions: {e}")
        return []


def process_question(question, answer_style):
    """
    Process a question (typed or follow-up):
      1. Append as a user message.
      2. Run the RAG workflow (via app.stream) and stream the assistant's response.
         If streaming produces no content (or errors occur), a fallback non-streaming
         approach is attempted.
    """
    # 1) Add user question to the chat
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    # Redirect stdout for debugging
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    assistant_response = ""

    # 2) Initialize empty assistant message for streaming the response
    st.session_state.messages.append({"role": "assistant", "content": ""})
    assistant_index = len(st.session_state.messages) - 1

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        debug_placeholder = st.empty()
        # CallBack handler get_streamlit_cb
        st_callback = get_streamlit_cb(st.empty())

        start_time = time.time()

        with st.spinner("Thinking..."):
            inputs = {
                "question": question,
                "hybrid_search": st.session_state.hybrid_search,
                "internet_search": st.session_state.internet_search,
                "answer_style": answer_style
            }
            try:
                # Attempt to stream response
                for idx, chunk in enumerate(app.stream(inputs, config={"callbacks": [st_callback]})):
                    debug_logs = output_buffer.getvalue()
                    debug_placeholder.text_area(
                        "Debug Logs", debug_logs, height=100, key=f"debug_logs_{idx}"
                    )
                    if "generate" in chunk and "generation" in chunk["generate"]:
                        assistant_response += chunk["generate"]["generation"]
                        styled_response = re.sub(
                            r'\[(.*?)\]',
                            r'<span class="reference">[\1]</span>',
                            assistant_response
                        )
                        response_placeholder.markdown(
                            f"**Assistant:** {styled_response}",
                            unsafe_allow_html=True
                        )
            except (tornado.websocket.WebSocketClosedError, tornado.iostream.StreamClosedError) as ws_error:
                # Log and silently handle known WebSocket errors without showing a modal.
                print(f"WebSocket connection closed: {ws_error}")
            except Exception as e:
                error_str = str(e)
                # Filter out non-critical errors (like "Bad message format") from showing in the UI.
                if "Bad message format" in error_str:
                    print(f"Non-critical error: {error_str}")
                else:
                    error_msg = f"Error generating response: {error_str}"
                    response_placeholder.error(error_msg)
                    st_callback.text = error_msg

            # If no response was produced by streaming, attempt fallback using invoke
            if not assistant_response.strip():
                try:
                    result = app.invoke(inputs)
                    if "generate" in result and "generation" in result["generate"]:
                        assistant_response = result["generate"]["generation"]
                        styled_response = re.sub(
                            r'\[(.*?)\]',
                            r'<span class="reference">[\1]</span>',
                            assistant_response
                        )
                        response_placeholder.markdown(
                            f"**Assistant:** {styled_response}",
                            unsafe_allow_html=True
                        )
                    else:
                        raise ValueError("No generation found in result")
                except Exception as fallback_error:
                    fallback_str = str(fallback_error)
                    if "Bad message format" in fallback_str:
                        print(f"Non-critical fallback error: {fallback_str}")
                    else:
                        print(f"Fallback also failed: {fallback_str}")
                        if not assistant_response.strip():
                            error_msg = ("Sorry, I encountered an error while generating a response. "
                                         "Please try again or select a different model.")
                            response_placeholder.error(error_msg)
                            assistant_response = error_msg

        # End timer and calculate generation time
        end_time = time.time()
        generation_time = end_time - start_time
        st.session_state["last_generation_time"] = generation_time

        # Optionally display the generation time if the timer is toggled on
        if st.session_state.get("show_timer", True):
            response_placeholder.markdown(
                f"*Generation time: {generation_time:.2f} seconds*")

        # Restore original stdout
        sys.stdout = sys.__stdout__

    # 3) Update the assistant message with the final response
    st.session_state.messages[assistant_index]["content"] = assistant_response
    st.session_state.followup_key += 1


# -------------------- Page Layout & Configuration --------------------
st.set_page_config(
    page_title="Smart Business Guide",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.markdown("""
    <style>
    .reference {
        color: blue;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    try:
        st.image("images/LOGO_UPBEAT.jpg", width=150, use_container_width=True)
    except Exception as e:
        st.warning("Unable to load image. Continuing without it.")

    st.title("🗣️ Smart Guide 1.0")
    st.markdown("**▶️ Actions:**")

    # Set default model selections if not present.
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    if "selected_routing_model" not in st.session_state:
        st.session_state.selected_routing_model = "gpt-4o"
    if "selected_grading_model" not in st.session_state:
        st.session_state.selected_grading_model = "gpt-4o"
    if "selected_embedding_model" not in st.session_state:
        st.session_state.selected_embedding_model = "text-embedding-3-large"

    model_list = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gpt-4o-mini",
        "gpt-4o",
        "deepseek-r1-distill-llama-70b"
    ]
    embed_list = [
        "text-embedding-3-large",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]

    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.selected_model = st.selectbox(
            "🤖 Select Answering LLM",
            model_list,
            key="model_selector",
            index=model_list.index(st.session_state.selected_model)
        )
        st.session_state.selected_routing_model = st.selectbox(
            "📡 Select Routing LLM",
            model_list,
            key="routing_model_selector",
            index=model_list.index(st.session_state.selected_routing_model)
        )
        st.session_state.selected_grading_model = st.selectbox(
            "🧮 Select Retrieval Grading LLM",
            model_list,
            key="grading_model_selector",
            index=model_list.index(st.session_state.selected_grading_model)
        )
        st.session_state.selected_embedding_model = st.selectbox(
            "🧠 Select Embedding Model",
            embed_list,
            key="embedding_model_selector",
            index=embed_list.index(st.session_state.selected_embedding_model)
        )
        answer_style = st.select_slider(
            "💬 Answer Style",
            options=["Concise", "Moderate", "Explanatory"],
            value="Explanatory",
            key="answer_style_slider"
        )
        st.session_state.answer_style = answer_style

    search_option = st.radio(
        "Search options",
        ["Reliable documents", "Reliable web sources",
            "Reliable docs & web sources"],
        index=0
    )
    st.session_state.hybrid_search = (
        search_option == "Hybrid search (Guides + internet)")
    st.session_state.internet_search = (
        search_option == "Internet search only")

    if st.button("🔄 Reset Conversation", key="reset_button"):
        st.session_state.messages = []

    # Toggle for displaying generation time
    st.checkbox("Show generation time", value=True, key="show_timer")
    # RAG workflow initilizate.
try:
    app = initialize_app(
        st.session_state.selected_model,
        st.session_state.selected_embedding_model,
        st.session_state.selected_routing_model,
        st.session_state.selected_grading_model,
        st.session_state.hybrid_search,
        st.session_state.internet_search,
        st.session_state.answer_style
    )
except Exception as e:
    st.error("Error initializing model, continuing with previous model: " + str(e))
    # (Optional) Initialize your primary LLM if needed.
    # st.session_state.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# -------------------- Main Title & Introduction --------------------
st.title("📘 Smart Guide for Entrepreneurship and Business Planning in Finland")
st.markdown(
    """
    <div style="text-align: left; font-size: 18px; margin-top: 20px; line-height: 1.6;">
        🤖 <b>Welcome to your Smart Business Guide!</b><br>
        I am here to assist you with:<br>
        <ul style="list-style-position: inside; text-align: left; display: inline-block;">
            <li>Finding answers from business and entrepreneurship guides in Finland</li>
            <li>Providing up-to-date information via AI-based internet search</li>
            <li>Automatically triggering search based on your query</li>
            <li>Specialized tools for tax-related info, permits, registrations, and more</li>
        </ul>
        <p style="margin-top: 10px;"><b>Start by typing your question in the chat below!</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- Display Conversation History --------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            # Process the response to add styling to references
            import re
            styled_response = re.sub(
                r'\[(.*?)\]',
                r'<span class="reference">[\1]</span>',
                message['content']
            )
            st.markdown(
                f"**Assistant:** {styled_response}",
                unsafe_allow_html=True
            )

# Display the last generation time outside the chat messages if enabled.
if st.session_state.get("show_timer", True) and "last_generation_time" in st.session_state:
    st.markdown(
        f"<small>Last Generation Time: {st.session_state.last_generation_time:.2f} seconds</small>", unsafe_allow_html=True)

# -------------------- Process a Pending Follow-Up (if any) --------------------
if st.session_state.pending_followup is not None:
    question = st.session_state.pending_followup
    st.session_state.pending_followup = None
    process_question(question, st.session_state.answer_style)

# -------------------- Process New User Input --------------------
user_input = st.chat_input("Type your question (Max. 200 char):")
if user_input:
    if len(user_input) > 200:
        st.error(
            "Your question exceeds 200 characters. Please shorten it and try again.")
    else:
        process_question(user_input, st.session_state.answer_style)

# -------------------- Helper function for Follow-Up --------------------


def handle_followup(question: str):
    st.session_state.pending_followup = question


# -------------------- Generate and Display Follow-Up Questions --------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    try:
        last_assistant_message = st.session_state.messages[-1]["content"]

        # Don't generate followup questions if response is empty or contains error messages
        if not last_assistant_message.strip() or "Sorry, I encountered an error" in last_assistant_message:
            st.session_state.followup_questions = []
        # Don't generate followup questions for unrelated responses
        elif "I apologize, but I'm designed to answer questions" in last_assistant_message:
            st.session_state.followup_questions = []
        else:
            # Get the last user message
            last_user_message = next(
                (msg["content"] for msg in reversed(st.session_state.messages)
                 if msg["role"] == "user"),
                ""
            )

            # Generate new questions only if the last assistant message has changed
            if st.session_state.last_assistant != last_assistant_message:
                print("Generating new followup questions")
                st.session_state.last_assistant = last_assistant_message
                try:
                    st.session_state.followup_questions = get_followup_questions(
                        last_user_message,
                        last_assistant_message
                    )
                except Exception as e:
                    print(f"Failed to generate followup questions: {e}")
                    st.session_state.followup_questions = []

        # Display follow-up questions only if we have valid ones
        if st.session_state.followup_questions and len(st.session_state.followup_questions) > 0:
            st.markdown("#### Related:")
            for i, question in enumerate(st.session_state.followup_questions):
                # Remove numbering e.g "1. ", "2. ", etc.
                clean_question = re.sub(r'^\d+\.\s*', '', question)
                st.button(
                    clean_question,
                    key=f"followup_{i}_{st.session_state.followup_key}",
                    on_click=handle_followup,
                    args=(clean_question,)
                )
    except Exception as e:
        print(f"Error in followup section: {e}")
        st.session_state.followup_questions = []
