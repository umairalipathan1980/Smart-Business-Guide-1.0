__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import sys
import io
import os
import time
import json
from agentic_rag import initialize_app
from langchain_openai import ChatOpenAI


# -------------------- Initialization --------------------
# Initialize session state variables if not already present.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "followup_key" not in st.session_state:
    st.session_state.followup_key = 0
if "selected_followup" not in st.session_state:
    st.session_state.selected_followup = None

# -------------------- Helper Functions --------------------
def get_followup_questions(last_user, last_assistant):
    """
    Generate three concise follow-up questions dynamically based on the latest conversation.
    Each question appears on its own line.
    """
    prompt = f"""Based on the conversation below:
        User: {last_user}
        Assistant: {last_assistant}

        Generate three concise follow-up questions that a user might ask next.
        Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question.
        Follow-up Questions:"""
    # Invoke the LLM to generate the follow-up questions.
    # response = st.session_state.followup_llm.invoke(prompt)
    response = st.session_state.llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    questions = [q.strip() for q in text.split('\n') if q.strip()]
    return questions[:3]

def process_question(question, answer_style):
    """
    Process a question (typed or follow-up) exactly as if it were entered by the user.
    This function:
      1. Appends the question as a user message.
      2. Invokes the full RAG workflow (via app.stream).
      3. Streams and appends the assistant's response.
      4. Displays debug stdout messages.
    """
    # Append the user message.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")
    
    # Capture and display debug output.
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    assistant_response = ""
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        debug_placeholder = st.empty()
        with st.spinner("Thinking..."):
            inputs = {
                "question": question,
                "hybrid_search": st.session_state.hybrid_search,
                "internet_search": st.session_state.internet_search,
                "answer_style": answer_style
            }
            for i, output in enumerate(app.stream(inputs)):
                debug_logs = output_buffer.getvalue()
                debug_placeholder.text_area("Debug Logs", debug_logs, height=100, key=f"debug_logs_{i}")
                if "generate" in output and "generation" in output["generate"]:
                    assistant_response += output["generate"]["generation"]
                    response_placeholder.markdown(f"**Assistant:** {assistant_response}")
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    sys.stdout = sys.__stdout__
    # Increment the counter so that follow-up button keys will be unique.
    st.session_state.followup_key += 1

# -------------------- Page Layout & Configuration --------------------
st.set_page_config(
    page_title="Smart Business Guide",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# -------------------- Sidebar --------------------
with st.sidebar:
    try:
        st.image("LOGO_UPBEAT.jpg", width=150, use_container_width=True)
    except Exception as e:
        st.warning("Unable to load image. Continuing without it.")

    st.title("🗣️ Smart Guide 1.0")
    st.markdown("**▶️ Actions:**")

    # Set default models if not present.
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    if "selected_routing_model" not in st.session_state:
        st.session_state.selected_routing_model = "gpt-4o"
    if "selected_grading_model" not in st.session_state:
        st.session_state.selected_grading_model = "gpt-4o"
    if "selected_embedding_model" not in st.session_state:
        st.session_state.selected_embedding_model = "text-embedding-3-large"
    # st.session_state.followup_llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.5)

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
            key="answer_style_slider",
            disabled=False
        )
        st.session_state.answer_style = answer_style

    search_option = st.radio(
        "Search options",
        ["Smart guide + tools", "Internet search only", "Hybrid search (Guides + internet)"],
        index=0
    )
    st.session_state.hybrid_search = (search_option == "Hybrid search (Guides + internet)")
    st.session_state.internet_search = (search_option == "Internet search only")
    
    reset_button = st.button("🔄 Reset Conversation", key="reset_button")
    if reset_button:
        st.session_state.messages = []
    
    # Initialize the RAG workflow.
    app = initialize_app(
        st.session_state.selected_model,
        st.session_state.selected_embedding_model,
        st.session_state.selected_routing_model,
        st.session_state.selected_grading_model,
        st.session_state.hybrid_search,
        st.session_state.internet_search,
        st.session_state.answer_style
    )

# -------------------- Process Selected Follow-Up (if any) --------------------
# This block is placed near the top so that if a follow-up has been clicked,
# it gets processed before the rest of the page is rendered.
if st.session_state.get("selected_followup") is not None:
    process_question(st.session_state.selected_followup, st.session_state.answer_style)
    st.session_state.selected_followup = None

# -------------------- Main Title & Introduction --------------------
st.title("📘 Smart Guide for Entrepreneurship and Business Planning in Finland")
st.markdown(
    """
    <div style="text-align: left; font-size: 18px; margin-top: 20px; line-height: 1.6;">
        🤖 <b>Welcome to your Smart Business Guide!</b><br>
        I am here to assist you with:<br>
        <ul style="list-style-position: inside; text-align: left; display: inline-block;">
            <li>AI agent-based approach for finding answers from business and entrepreneurship guides in Finland</li>
            <li>Providing up-to-date information through AI-based internet search</li>
            <li>Automatically invoking AI-based internet search based on query understanding</li>
            <li>Specialized tools for tax-related information, permits & licenses, business registration, residence permits, etc.</li>
        </ul>
        <p style="margin-top: 10px;"><b>Start by typing your question in the chat below, and I'll provide tailored answers for your business needs!</b></p>
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
            st.markdown(f"**Assistant:** {message['content']}")

# -------------------- Process New User Input --------------------
if user_input := st.chat_input("Type your question (Max. 200 char):"):
    if len(user_input) > 200:
        st.error("Your question exceeds 200 characters. Please shorten it and try again.")
    else:
        process_question(user_input, st.session_state.answer_style)

# -------------------- Generate and Display Follow-Up Questions --------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    # Get the last user message.
    last_user_message = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break
    last_assistant_message = st.session_state.messages[-1]["content"]
    followup_questions = get_followup_questions(last_user_message, last_assistant_message)
    st.markdown("#### Related:")
    for i, question in enumerate(followup_questions):
        if st.button(question, key=f"followup_{i}_{st.session_state.followup_key}"):
            st.session_state.selected_followup = question
            st.rerun()  # Use experimental_rerun() here as well.


