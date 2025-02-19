# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import io
import re  # ensure regex is imported
import sys

import streamlit as st
import torch
from langchain_openai import ChatOpenAI

from agentic_rag import initialize_app
from st_callback import get_streamlit_cb

# Fix for "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!"
torch.classes.__path__ = []
# TODO: Bug fix App crashes when selecting "llama-3.1-8b-instant" model.

hardcoded_questions = [
    "Miten rekisteröin toiminimen Suomessa?",
    "Mitä eri yritysmuotoja on olemassa?",
    "Miten hankin alkupääoman yritykselle?"
]

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
    try:
        prompt = f"""Based on the conversation below:
            User: {last_user}
            Assistant: {last_assistant}
            Generate three concise follow-up questions that a user might ask next.
            Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question. Focus on brevity.
            Follow-up Questions:"""
        response = st.session_state.llm.invoke(prompt)
        text = response.content if hasattr(
            response, "content") else str(response)
        questions = [q.strip() for q in text.split('\n') if q.strip()]
        return questions[:3]
    except Exception as e:
        print(f"Failed to generate follow-up questions: {e}")
        return []

def process_question(question, answer_style):
    # 1) Add user question to the chat
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    # Redirect stdout for debugging
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    assistant_response = ""

    # 2) Initialize empty assistant message where the response will be streamed
    st.session_state.messages.append({"role": "assistant", "content": ""})
    assistant_index = len(st.session_state.messages) - 1

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        debug_placeholder = st.empty()
        st_callback = get_streamlit_cb(st.empty())

        with st.spinner("Thinking..."):
            try:
                inputs = {
                    "question": question,
                    "hybrid_search": st.session_state.hybrid_search,
                    "internet_search": st.session_state.internet_search,
                    "answer_style": answer_style
                }

                # Process the question
                for idx, chunk in enumerate(app.stream(inputs, config={"callbacks": [st_callback]})):
                    debug_logs = output_buffer.getvalue()
                    debug_placeholder.text_area(
                        "Debug Logs", debug_logs, height=100, key=f"debug_logs_{idx}")
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

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                response_placeholder.error(error_msg)
                st_callback.text = error_msg

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

    # Initialize your RAG workflow here.
    app = initialize_app(
        st.session_state.selected_model,
        st.session_state.selected_embedding_model,
        st.session_state.selected_routing_model,
        st.session_state.selected_grading_model,
        st.session_state.hybrid_search,
        st.session_state.internet_search,
        st.session_state.answer_style
    )

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
        st.rerun()

# -------------------- Helper function for Follow-Up --------------------


def handle_followup(question: str):
    st.session_state.pending_followup = question
    st.rerun()


# -------------------- Generate and Display Follow-Up Questions --------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    try:
        last_assistant_message = st.session_state.messages[-1]["content"]
        last_user_message = next(
            (msg["content"] for msg in reversed(st.session_state.messages)
             if msg["role"] == "user"),
            ""
        )

        # If the last assistant message is the "unrelated" response, do not generate follow-up questions
        if "I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland." in last_assistant_message:
            st.session_state.followup_questions = []
        else:
            # Generate new questions if the last assistant message has changed
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
                    # Reset followup questions on error
                    st.session_state.followup_questions = []

        # Display follow-up question buttons only if list is not empty
        if st.session_state.followup_questions and len(st.session_state.followup_questions) > 0:
            st.markdown("#### Related:")
            for i, question in enumerate(st.session_state.followup_questions):
                # Remove numbering e.g "1. "
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
