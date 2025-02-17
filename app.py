
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import io
import sys
import time

import streamlit as st
import torch
from langchain_openai import ChatOpenAI

from agentic_rag import initialize_app
from st_callback import get_streamlit_cb

torch.classes.__path__ = [] # Fix for "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!"
# TODO: Bug fix App crashes when selecting "llama-3.1-8b-instant" model.

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
    response = st.session_state.llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    questions = [q.strip() for q in text.split('\n') if q.strip()]
    return questions[:3]

def process_question(question, answer_style):
    """
    Processes a question with streaming support.
    1. Appends the user's message.
    2. Streams the assistant's response using get_streamlit_cb.
    3. Updates the assistant's message in session_state with the final response.
    """
    # Append the user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    # Create an empty assistant message if the previous one isn't already empty
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == "":
        assistant_index = len(st.session_state.messages) - 1
    else:
        st.session_state.messages.append({"role": "assistant", "content": ""})
        assistant_index = len(st.session_state.messages) - 1

    with st.chat_message("assistant"):
        stream_container = st.empty()  # Container to display streaming output
        st_callback = get_streamlit_cb(stream_container)
        debug_placeholder = st.empty()

        with st.spinner("Thinking..."):
            try:
                inputs = {
                    "question": question,
                    "hybrid_search": st.session_state.hybrid_search,
                    "internet_search": st.session_state.internet_search,
                    "answer_style": answer_style
                }

                # Redirect stdout to capture debug logs
                output_buffer = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = output_buffer

                try:
                    # Correct enumeration: use idx and chunk
                    for idx, chunk in enumerate(app.stream(inputs, config={"callbacks": [st_callback]})):
                        debug_logs = output_buffer.getvalue()
                        debug_placeholder.text_area(
                            "Debug Logs",
                            debug_logs,
                            height=100,
                            key=f"debug_logs_{idx}"
                        )
                finally:
                    # Always restore stdout
                    sys.stdout = old_stdout

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                stream_container.error(error_msg)
                st_callback.text = error_msg

        # Retrieve the final response from the callback
        final_response = st_callback.text
        stream_container.markdown(f"**Assistant:** {final_response}", unsafe_allow_html=True)

    # Update the assistant message in session_state
    st.session_state.messages[assistant_index]["content"] = final_response
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
        ["Reliable documents", "Reliable web sources", "Reliable docs & web sources"],
        index=0
    )
    st.session_state.hybrid_search = (search_option == "Hybrid search (Guides + internet)")
    st.session_state.internet_search = (search_option == "Internet search only")
    
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

# -------------------- Process a Pending Follow-Up (if any) --------------------
# Now that `app` is defined, process any pending follow-up.
if st.session_state.pending_followup is not None:
    question = st.session_state.pending_followup
    st.session_state.pending_followup = None  # Clear to ensure one-time processing.
    process_question(question, st.session_state.answer_style)
    st.rerun()  # Rerun the app to update the conversation state.

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

# -------------------- Process New User Input --------------------
user_input = st.chat_input("Type your question (Max. 200 char):")
if user_input:
    if len(user_input) > 200:
        st.error("Your question exceeds 200 characters. Please shorten it and try again.")
    else:
        process_question(user_input, st.session_state.answer_style)
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
                st.session_state.followup_questions = get_followup_questions(
                    last_user_message, 
                    last_assistant_message
                )
        
        # Display follow-up question buttons if any are available
        if st.session_state.followup_questions:
            st.markdown("#### Related:")
            for i, question in enumerate(st.session_state.followup_questions):
                button_key = f"followup_{i}_{st.session_state.followup_key}"
                if st.button(question, key=button_key):
                    print(f"Followup button {i} clicked: {question}")
                    # Set pending follow-up without triggering an immediate rerun
                    st.session_state.pending_followup = question
                    # Do NOT call st.rerun() here
    except Exception as e:
        print(f"Error in followup section: {e}")












# import streamlit as st
# import sys
# import io
# import time
# from agentic_rag import initialize_app
# from langchain_openai import ChatOpenAI

# # -------------------- Initialization --------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "followup_key" not in st.session_state:
#     st.session_state.followup_key = 0
# if "pending_followup" not in st.session_state:
#     st.session_state.pending_followup = None
# if "last_assistant" not in st.session_state:
#     st.session_state.last_assistant = None
# if "followup_questions" not in st.session_state:
#     st.session_state.followup_questions = []

# # -------------------- Helper Functions --------------------
# def get_followup_questions(last_user, last_assistant):
#     """
#     Generate three concise follow-up questions dynamically based on the latest conversation.
#     """
#     prompt = f"""Based on the conversation below:
#         User: {last_user}
#         Assistant: {last_assistant}
#         Generate three concise follow-up questions that a user might ask next.
#         Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question. Focus on brevity.
#         Follow-up Questions:"""
#     response = st.session_state.llm.invoke(prompt)
#     text = response.content if hasattr(response, "content") else str(response)
#     questions = [q.strip() for q in text.split('\n') if q.strip()]
#     return questions[:3]

# def process_question(question, answer_style):
#     """
#     Process a question (typed or follow-up):
#       1. Append as a user message.
#       2. Run the RAG workflow (via app.stream).
#       3. Stream and append the assistant's response.
#     """
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(f"**You:** {question}")
    
#     output_buffer = io.StringIO()
#     sys.stdout = output_buffer
#     assistant_response = ""
#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
#         debug_placeholder = st.empty()
#         with st.spinner("Thinking..."):
#             inputs = {
#                 "question": question,
#                 "hybrid_search": st.session_state.hybrid_search,
#                 "internet_search": st.session_state.internet_search,
#                 "answer_style": answer_style
#             }
#             for i, output in enumerate(app.stream(inputs)):
#                 debug_logs = output_buffer.getvalue()
#                 debug_placeholder.text_area("Debug Logs", debug_logs, height=100, key=f"debug_logs_{i}")
#                 if "generate" in output and "generation" in output["generate"]:
#                     assistant_response += output["generate"]["generation"]
#                     response_placeholder.markdown(f"**Assistant:** {assistant_response}")
#     st.session_state.messages.append({"role": "assistant", "content": assistant_response})
#     sys.stdout = sys.__stdout__
#     st.session_state.followup_key += 1

# # -------------------- Page Layout & Configuration --------------------
# st.set_page_config(
#     page_title="Smart Business Guide",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🧠"
# )

# # -------------------- Sidebar --------------------
# with st.sidebar:
#     try:
#         st.image("images/LOGO_UPBEAT.jpg", width=150, use_container_width=True)
#     except Exception as e:
#         st.warning("Unable to load image. Continuing without it.")

#     st.title("🗣️ Smart Guide 1.0")
#     st.markdown("**▶️ Actions:**")

#     # Set default model selections if not present.
#     if "selected_model" not in st.session_state:
#         st.session_state.selected_model = "gpt-4o"
#     if "selected_routing_model" not in st.session_state:
#         st.session_state.selected_routing_model = "gpt-4o"
#     if "selected_grading_model" not in st.session_state:
#         st.session_state.selected_grading_model = "gpt-4o"
#     if "selected_embedding_model" not in st.session_state:
#         st.session_state.selected_embedding_model = "text-embedding-3-large"

#     model_list = [
#         "llama-3.1-8b-instant",
#         "llama-3.3-70b-versatile",
#         "llama3-70b-8192",
#         "llama3-8b-8192",
#         "mixtral-8x7b-32768",
#         "gemma2-9b-it",
#         "gpt-4o-mini",
#         "gpt-4o",
#         "deepseek-r1-distill-llama-70b"
#     ]
#     embed_list = [
#         "text-embedding-3-large",
#         "sentence-transformers/all-MiniLM-L6-v2"
#     ]

#     with st.expander("⚙️ Settings", expanded=False):
#         st.session_state.selected_model = st.selectbox(
#             "🤖 Select Answering LLM",
#             model_list,
#             key="model_selector",
#             index=model_list.index(st.session_state.selected_model)
#         )
#         st.session_state.selected_routing_model = st.selectbox(
#             "📡 Select Routing LLM",
#             model_list,
#             key="routing_model_selector",
#             index=model_list.index(st.session_state.selected_routing_model)
#         )
#         st.session_state.selected_grading_model = st.selectbox(
#             "🧮 Select Retrieval Grading LLM",
#             model_list,
#             key="grading_model_selector",
#             index=model_list.index(st.session_state.selected_grading_model)
#         )
#         st.session_state.selected_embedding_model = st.selectbox(
#             "🧠 Select Embedding Model",
#             embed_list,
#             key="embedding_model_selector",
#             index=embed_list.index(st.session_state.selected_embedding_model)
#         )
#         answer_style = st.select_slider(
#             "💬 Answer Style",
#             options=["Concise", "Moderate", "Explanatory"],
#             value="Explanatory",
#             key="answer_style_slider"
#         )
#         st.session_state.answer_style = answer_style

#     search_option = st.radio(
#         "Search options",
#         ["Smart guide + tools", "Internet search only", "Hybrid search (Guides + internet)"],
#         index=0
#     )
#     st.session_state.hybrid_search = (search_option == "Hybrid search (Guides + internet)")
#     st.session_state.internet_search = (search_option == "Internet search only")
    
#     if st.button("🔄 Reset Conversation", key="reset_button"):
#         st.session_state.messages = []

#     # Initialize your RAG workflow here.
#     app = initialize_app(
#         st.session_state.selected_model,
#         st.session_state.selected_embedding_model,
#         st.session_state.selected_routing_model,
#         st.session_state.selected_grading_model,
#         st.session_state.hybrid_search,
#         st.session_state.internet_search,
#         st.session_state.answer_style
#     )
    
#     # (Optional) Initialize your primary LLM if needed.
#     # st.session_state.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# # -------------------- Process a Pending Follow-Up (if any) --------------------
# # Now that `app` is defined, process any pending follow-up.
# if st.session_state.pending_followup is not None:
#     question = st.session_state.pending_followup
#     st.session_state.pending_followup = None  # Clear to ensure one-time processing.
#     process_question(question, st.session_state.answer_style)
#     st.rerun()  # Rerun the app to update the conversation state.

# # -------------------- Main Title & Introduction --------------------
# st.title("📘 Smart Guide for Entrepreneurship and Business Planning in Finland")
# st.markdown(
#     """
#     <div style="text-align: left; font-size: 18px; margin-top: 20px; line-height: 1.6;">
#         🤖 <b>Welcome to your Smart Business Guide!</b><br>
#         I am here to assist you with:<br>
#         <ul style="list-style-position: inside; text-align: left; display: inline-block;">
#             <li>Finding answers from business and entrepreneurship guides in Finland</li>
#             <li>Providing up-to-date information via AI-based internet search</li>
#             <li>Automatically triggering search based on your query</li>
#             <li>Specialized tools for tax-related info, permits, registrations, and more</li>
#         </ul>
#         <p style="margin-top: 10px;"><b>Start by typing your question in the chat below!</b></p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # -------------------- Display Conversation History --------------------
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         with st.chat_message("user"):
#             st.markdown(f"**You:** {message['content']}")
#     elif message["role"] == "assistant":
#         with st.chat_message("assistant"):
#             st.markdown(f"**Assistant:** {message['content']}")

# # -------------------- Process New User Input --------------------
# user_input = st.chat_input("Type your question (Max. 200 char):")
# if user_input:
#     if len(user_input) > 200:
#         st.error("Your question exceeds 200 characters. Please shorten it and try again.")
#     else:
#         process_question(user_input, st.session_state.answer_style)
#         st.rerun()

# # -------------------- Generate and Display Follow-Up Questions --------------------
# if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
#     last_assistant_message = st.session_state.messages[-1]["content"]
#     last_user_message = next(
#         (msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"),
#         ""
#     )
#     if st.session_state.last_assistant != last_assistant_message:
#         st.session_state.last_assistant = last_assistant_message
#         st.session_state.followup_questions = get_followup_questions(last_user_message, last_assistant_message)
    
#     st.markdown("#### Related:")
#     for i, question in enumerate(st.session_state.followup_questions):
#         if st.button(question, key=f"followup_{i}_{st.session_state.followup_key}"):
#             st.session_state.pending_followup = question
#             st.rerun()





































