
import logging
import os
import re
import sys
import warnings
from typing import List

import requests
import spacy
import streamlit as st
from bs4 import BeautifulSoup
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_community.document_loaders import (UnstructuredMarkdownLoader,
                                                  WebBaseLoader)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from tavily import TavilyClient
from typing_extensions import TypedDict

# Set up environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["USER_AGENT"] = "AgenticRAG/1.0"
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Resolve or suppress warnings
# Set global logging level to ERROR
logging.basicConfig(level=logging.ERROR, force=True)
# Suppress all SageMaker logs
logging.getLogger("sagemaker").setLevel(logging.CRITICAL)
logging.getLogger("sagemaker.config").setLevel(logging.CRITICAL)

# Ignore the specific FutureWarning from Hugging Face Transformers
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning
)
# General suppression for other warnings (optional)
warnings.filterwarnings("ignore")
# Configure logging
logging.basicConfig(level=logging.INFO)
###################################################

# Define paths and parameters
data_file_path = 'Becoming an entrepreneur in Finland.md'
DATA_FOLDER = 'data'
persist_directory_openai = 'data/chroma_db_llamaparse-openai'
persist_directory_huggingface = 'data/chroma_db_llamaparse-huggincface'
collection_name = 'rag'
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200


def remove_tags(soup):
    # Remove unwanted tags
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        element.decompose()

    # Extract text while preserving structure
    content = ""
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = element.get_text(strip=True)
        if element.name.startswith('h'):
            level = int(element.name[1])
            content += '#' * level + ' ' + text + '\n\n'  # Markdown-style headings
        elif element.name == 'p':
            content += text + '\n\n'
        elif element.name == 'li':
            content += '- ' + text + '\n'
    return content

# @st.cache_data


def get_info(URLs):
    """
    Fetch and return contact information from predefined URLs.
    """
    combined_info = ""
    for url in URLs:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                combined_info += "URL: " + url + \
                    ": " + remove_tags(soup) + "\n\n"
            else:
                combined_info += f"Failed to retrieve information from {url}\n\n"
        except Exception as e:
            combined_info += f"Error fetching URL {url}: {e}\n\n"
    return combined_info

# @st.cache_data


def staticChunker(folder_path):
    docs = []
    print(
        f"Creating chunks. CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")

    # Loop through all .md files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            # Load documents from the Markdown file
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()

            # Add file-specific metadata (optional)
            for doc in documents:
                doc.metadata["source_file"] = file_name

            # Split loaded documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunked_docs = text_splitter.split_documents(documents)
            docs.extend(chunked_docs)
    return docs

# @st.cache_resource


def load_or_create_vs(persist_directory):
    # Check if the vector store directory exists
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=st.session_state.embed_model,
            collection_name=collection_name
        )
    else:
        print("Vector store not found. Creating a new one...\n")
        docs = staticChunker(DATA_FOLDER)
        print("Computing embeddings...")
        # Create and persist a new Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=st.session_state.embed_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print('Vector store created and persisted successfully!')

    return vectorstore


def initialize_app(model_name, selected_embedding_model, selected_routing_model, selected_grading_model, hybrid_search, internet_search, answer_style):
    """
    Initialize embeddings, vectorstore, retriever, and LLM for the RAG workflow.
    Reinitialize components only if the selection has changed.
    """
    # Track current state to prevent redundant initialization
    if "current_model_state" not in st.session_state:
        st.session_state.current_model_state = {
            "answering_model": None,
            "embedding_model": None,
            "routing_model": None,
            "grading_model": None,
        }

    # Check if models or settings have changed
    state_changed = (
        st.session_state.current_model_state["answering_model"] != model_name or
        st.session_state.current_model_state["embedding_model"] != selected_embedding_model or
        st.session_state.current_model_state["routing_model"] != selected_routing_model or
        st.session_state.current_model_state["grading_model"] != selected_grading_model
    )

    # Reinitialize components only if settings have changed
    if state_changed:
        try:
            st.session_state.embed_model = initialize_embedding_model(
                selected_embedding_model)

            # Update vectorstore
            persist_directory = persist_directory_openai if "text-" in selected_embedding_model else persist_directory_huggingface
            st.session_state.vectorstore = load_or_create_vs(persist_directory)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 5})

            st.session_state.llm = initialize_llm(model_name, answer_style)
            st.session_state.router_llm = initialize_router_llm(
                selected_routing_model)
            st.session_state.grader_llm = initialize_grading_llm(
                selected_grading_model)
            st.session_state.doc_grader = initialize_grader_chain()

            # Save updated state
            st.session_state.current_model_state.update({
                "answering_model": model_name,
                "embedding_model": selected_embedding_model,
                "routing_model": selected_routing_model,
                "grading_model": selected_grading_model,
            })
        except Exception as e:
            st.error(f"Error during initialization: {e}")
            # Restore previous state if available
            if st.session_state.current_model_state["answering_model"]:
                st.warning(f"Continuing with previous configuration")
            else:
                # Fallback to OpenAI if no previous state
                st.session_state.llm = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0.0, streaming=True)
                st.session_state.router_llm = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0.0)
                st.session_state.grader_llm = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0.0)

    print(f"Using LLM: {model_name}, Router LLM: {selected_routing_model}, Grader LLM:{selected_grading_model}, embedding model: {selected_embedding_model}")

    try:
        return workflow.compile()
    except Exception as e:
        st.error(f"Error compiling workflow: {e}")
        # Return a simple dummy workflow that just echoes the input
        return lambda x: {"generation": "Error in workflow. Please try a different model.", "question": x.get("question", "")}
# @st.cache_resource


def initialize_llm(model_name, answer_style):
    if "llm" not in st.session_state or st.session_state.llm.model_name != model_name:
        if answer_style == "Concise":
            temperature = 0.0
        elif answer_style == "Moderate":
            temperature = 0.0
        elif answer_style == "Explanatory":
            temperature = 0.0

        if "gpt-" in model_name:
            st.session_state.llm = ChatOpenAI(
                model=model_name, temperature=temperature, streaming=True)
        elif "deepseek-" in model_name:
            # Deepseek models need "hidden" reasoning_format to prevent <think> tags that otherwise cause issues
            st.session_state.llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                streaming=True,
                model_kwargs={"reasoning_format": "hidden"}
            )
        else:
            st.session_state.llm = ChatGroq(
                model=model_name, temperature=temperature, streaming=True)

    return st.session_state.llm


def initialize_embedding_model(selected_embedding_model):
    # Check if the embed_model exists in session_state
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = None

    # Check if the current model matches the selected one
    current_model_name = None
    if st.session_state.embed_model:
        if hasattr(st.session_state.embed_model, "model"):
            current_model_name = st.session_state.embed_model.model
        elif hasattr(st.session_state.embed_model, "model_name"):
            current_model_name = st.session_state.embed_model.model_name

    # Initialize a new model if it doesn't match the selected one
    if current_model_name != selected_embedding_model:
        if "text-" in selected_embedding_model:
            st.session_state.embed_model = OpenAIEmbeddings(
                model=selected_embedding_model)
        else:
            st.session_state.embed_model = HuggingFaceEmbeddings(
                model_name=selected_embedding_model)

    return st.session_state.embed_model

# @st.cache_resource

# FIX: mixtral model won't work with ChatGroq idk why. Maybe add gpt-4o-mini as fallback


def initialize_router_llm(selected_routing_model):
    if "router_llm" not in st.session_state or st.session_state.router_llm.model_name != selected_routing_model:
        if "gpt-" in selected_routing_model:
            st.session_state.router_llm = ChatOpenAI(
                model=selected_routing_model, temperature=0.0)
        elif "deepseek-" in selected_routing_model:
            st.session_state.router_llm = ChatGroq(
                model=selected_routing_model,
                temperature=0.0,
                model_kwargs={"reasoning_format": "hidden"}
            )
        # Uncomment this block to use gpt-4o-mini as a fallback for mixtral models. Because 20.2.2025 mixtral model won't in router_llm
        # elif "mixtral" in selected_routing_model.lower():
        #     st.session_state.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            st.session_state.router_llm = ChatGroq(
                model=selected_routing_model, temperature=0.0)

    return st.session_state.router_llm

# @st.cache_resource


def initialize_grading_llm(selected_grading_model):
    if "grader_llm" not in st.session_state or st.session_state.grader_llm.model_name != selected_grading_model:
        if "gpt-" in selected_grading_model:
            st.session_state.grader_llm = ChatOpenAI(
                model=selected_grading_model, temperature=0.0, max_tokens=16000)
        elif "deepseek-" in selected_grading_model:
            # Deepseek-models need "hidden" reasoning_format to prevent <think> tags from leaking
            st.session_state.grader_llm = ChatGroq(
                model=selected_grading_model,
                temperature=0.0,
                model_kwargs={"reasoning_format": "hidden"}
            )
        else:
            st.session_state.grader_llm = ChatGroq(
                model=selected_grading_model, temperature=0.0)

    return st.session_state.grader_llm


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

rag_prompt = PromptTemplate(
    template=r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a helpful, highly accurate and trustworthy assistant specialized in answering questions related to business, entrepreneurship, and the realted matters in Finland. 
                Your responses must strictly adhere to the provided context, answer style, and question's language using the follow rules:

                1. **Question and answer language**: 
                - Detect the question's language (e.g., English, Finnish, Russian, Estonian, Arabic) and answer in the same language. The context could be in English or any other languge. 
                - **very important**: make sure that your response is in the same language as the question's. 
                2. If the context documents contain 'Internt search results' in 'page_content' field, always consider them in your response. 

                3. **Context-Only Answers with a given answer style**:
                - Always base your answers on the provided context and answer style.
                - If the context does not contain relevant information, respond with: 'No information found. Try rephrasing the query.'
                - If the context contains some pieces of the required information, answer with that information and very briefly mention that the answer to other parts could not be found.
                - If the context explicitly states 'I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland,' output this context verbatim.

                4. **Response style**:
                - Address the query directly without unnecessary or speculative information.
                - Do not draw from your knowledge base; strictly use the given context. However, take some liberty to provide more explanations and illustrations for better clarity and demonstration from your knowledge and experience only if answer style is "Moderate" or "Explanatory". 
                5. **Answer style**
                - If answer style = "Concise", generate a concise answer. 
                - If answer style = "Moderate", use a moderate approach to generate answer where you can provide a little bit more explanation and elaborate the answer to improve clarity, integrating your own experience. 
                - If answer style = "Explanatory", provide a detailed and elaborated answer by providing more explanations with examples and illustrations to improve clarity in best possible way, integrating your own experience. However, the explanations, examples and illustrations should be strictly based on the context. 

                6. **Conversational tone**
                 - Maintain a conversational and helping style which should tend to guide the user and provide him help, hints and offers to further help and information. 
                 - Use simple language. Explain difficult concepts or terms wherever needed. Present the information in the best readable form.

                7. **Formatting Guidelines**:
                - Use bullet points for lists.
                - Include line breaks between sections for clarity.
                - Highlight important numbers, dates, and terms using **bold** formatting.
                - Create tables wherever appropriate to present data clearly.
                - If there are discrepancies in the context, clearly explain them.

                8. **Citation Rules**:
                - **very important**: Include citations in the answer at all relevant places if they are present in the context. Under no circumstances ignore them. 
                - For responses based on the context documents containing 'page_content' field of 'Smart guide results:', cite the document name and page number with each piece of information in the format: [document_name, page xx].
                - For responses based on the documents containing the 'page_content' field of 'Internet search results:', include all the URLs in hyperlink form returned by the websearch. **very important**: The URLs should be labelled with the website name. 
                - Do not invent any citation or URL. Only use the citation or URL in the context. 

                9. **Hybrid Context Handling**:
                - If and only if the context contains documents with two different 'page_content' with the names 'Smart guide results:' and 'Internet search results:', structure your response in corresponding sections with the following headings:
                    - **Smart guide results**: Include data from 'Smart guide results' and its citations in the format: [document_name, page xx]. If the 'Smart guide results' do not contain any information relevant to the question, output 'No information found' in this section.
                    - **Internet search results**: Include data from 'Internet search results' and its citations (URLs). 
                    - Ensure that you create two separate sections if and only if the context contain documents with two different 'page_content': 'Smart guide results:' and 'Internet search results:'.
                    - Do not create two different sections or mention 'Smart guide results:' or 'Internet search results:' in your response if the context does not contain documents with two different 'page_content': 'Smart guide results:' and 'Internet search results:'.
                    - Always include the document with 'page_content' of 'Internet search results:' in your response.
                    - If answer style = "Explanatory", both the sections should be detailed and should contain all the points relevant to the question.
                10. **Integrity and Trustworthiness**:
                - Ensure every part of your response complies with these rules.

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: {question} 
                Context: {context} 
                Answer style: {answer_style}
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "answer_style"]

)


def initialize_grader_chain():
    # Data model for LLM output format
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM for grading
    structured_llm_grader = st.session_state.grader_llm.with_structured_output(
        GradeDocuments)

    # Prompt template for grading
    SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.

    Follow these instructions for grading:
    - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    - Your grade should be either 'Yes' or 'No' to indicate whether the document is relevant to the question or not."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
    {documents}
    User question:
    {question}
    """),
    ])

    # Build grader chain
    return grade_prompt | structured_llm_grader


def grade_documents(state):
    question = state["question"]
    documents = state.get("documents", [])
    filtered_docs = []

    if not documents:
        print("No documents retrieved for grading.")
        return {"documents": [], "question": question, "web_search_needed": "Yes"}

    print(
        f"Grading retrieved documents with {st.session_state.grader_llm.model_name}")

    for count, doc in enumerate(documents):
        try:
            # Evaluate document relevance
            score = st.session_state.doc_grader.invoke(
                {"documents": [doc], "question": question})
            print(f"Chunk {count} relevance: {score}")
            if score.binary_score == "Yes":
                filtered_docs.append(doc)
        except Exception as e:
            print(f"Error grading document chunk {count}: {e}")

    web_search_needed = "Yes" if not filtered_docs else "No"
    return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}


def route_after_grading(state):
    web_search_needed = state.get("web_search_needed", "No")
    print(f"Routing decision based on web_search_needed={web_search_needed}")
    if web_search_needed == "Yes":
        return "websearch"
    else:
        return "generate"

# Define graph state class


class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]
    answer_style: str


def retrieve(state):
    print("Retrieving documents")
    question = state["question"]
    documents = st.session_state.retriever.invoke(question)
    return {"documents": documents, "question": question}


def format_documents(documents):
    """Format documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in documents)


def generate(state):
    question = state["question"]
    documents = state.get("documents", [])
    answer_style = state.get("answer_style", "Concise")

    if "llm" not in st.session_state:
        st.session_state.llm = initialize_llm(
            st.session_state.selected_model, answer_style)

    rag_chain = rag_prompt | st.session_state.llm | StrOutputParser()

    if not documents:
        print("No documents available for generation.")
        return {"generation": "No relevant documents found.", "documents": documents, "question": question}

    tried_models = set()
    original_model = st.session_state.selected_model
    current_model = original_model

    while len(tried_models) < len(model_list):
        try:
            tried_models.add(current_model)
            st.session_state.llm = initialize_llm(current_model, answer_style)
            rag_chain = rag_prompt | st.session_state.llm | StrOutputParser()

            # context = format_documents(documents)
            context = documents
            generation = rag_chain.invoke(
                {"context": context, "question": question, "answer_style": answer_style})

            print(f"Generating a {answer_style} length response.")
            # print(f"Response generated with {st.session_state.llm.model_name} model.")
            print("Done.")

            if current_model != original_model:
                print(f"Reverting to original model: {original_model}")
                st.session_state.llm = initialize_llm(
                    original_model, answer_style)

            return {"documents": documents, "question": question, "generation": generation}

        except Exception as e:
            error_message = str(e)
            if "rate_limit_exceeded" in error_message or "Request too large" in error_message or "Please reduce the length of the messages or completion" in error_message:
                print(f"Model's rate limit exceeded or request too large.")
                current_model = model_list[(model_list.index(
                    current_model) + 1) % len(model_list)]
                print(f"Switching to model: {current_model}")
            else:
                return {
                    "generation": f"Error during generation: {error_message}",
                    "documents": documents,
                    "question": question,
                }

    return {
        "generation": "Unable to process the request due to limitations across all models.",
        "documents": documents,
        "question": question,
    }


def handle_unrelated(state):
    question = state["question"]
    documents = state.get("documents", [])
    response = "I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland. Could you please rephrase your question to focus on these topics?"
    documents.append(Document(page_content=response))
    return {"generation": response, "documents": documents, "question": question}


def hybrid_search(state):
    question = state["question"]
    print("Invoking retriever...")
    vector_docs = st.session_state.retriever.invoke(question)
    web_docs = web_search({"question": question})["documents"]

    # Add headings to distinguish between vector and web search results
    vector_results = [Document(
        page_content="Smart guide results:" + doc.page_content) for doc in vector_docs]

    # Check if any web_docs already contain "Internet search results:"
    web_results_contain_header = any(
        "Internet search results:" in doc.page_content for doc in web_docs)

    # Add "Internet search results:" only if not already present in any web doc
    if not web_results_contain_header:
        web_results = [
            Document(page_content="Internet search results:" + doc.page_content) for doc in web_docs
        ]
    else:
        web_results = web_docs  # Keep web_docs unchanged if they already contain the header

    combined_docs = vector_results + web_results
    return {"documents": combined_docs, "question": question}


def web_search(state):
    if "tavily_client" not in st.session_state:
        st.session_state.tavily_client = TavilyClient()
    question = state["question"]
    question = re.sub(r'\b\w+\\|Internet search\b', '', question).strip()
    question = question + " in Finland"
    documents = state.get("documents", [])
    try:
        print("Invoking internet search...")
        search_result = st.session_state.tavily_client.get_search_context(
            query=question,
            # can be switched to 'advanced' mode that requires 2 credits per search.
            search_depth="basic",
            max_tokens=4000,
            max_results=10,
            include_domains=[
                "migri.fi",
                "enterfinland.fi",
                "businessfinland.fi",
                "kela.fi",
                "vero.fi",
                "suomi.fi",
                "valvira.fi",
                "finlex.fi",
                "hus.fi",
                "lvm.fi",
                "thefinlandbusinesspress.fi",
                "infofinland.fi",
                "ely-keskus.fi",
                "yritystulkki.fi",
                "tem.fi",
                "prh.fi"
            ],
        )
        # Handle different types of results
        if isinstance(search_result, str):
            web_results = search_result
        elif isinstance(search_result, dict) and "documents" in search_result:
            web_results = "Internet search results:".join(
                [doc.get("content", "") for doc in search_result["documents"]])
        else:
            web_results = "No valid results returned by TavilyClient."
        web_results_doc = Document(page_content=web_results)
        documents.append(web_results_doc)
    except Exception as e:
        print(f"Error during web search: {e}")
        # Ensure workflow can continue gracefully
        documents.append(Document(page_content=f"Web search failed: {e}"))
    return {"documents": documents, "question": question}


# # Router function
def route_question(state):
    question = state["question"]
    hybrid_search_enabled = state.get("hybrid_search", False)
    internet_search_enabled = state.get("internet_search", False)

    if hybrid_search_enabled:
        return "hybrid_search"

    if internet_search_enabled:
        return "websearch"

    tool_selection = {
        "websearch": (
            "Questions requiring current statistics or real-time information such as tax rate, taxation rules, taxable incomes, tax exemptions, the tax filing process, immigration or visa process or questions related to Finnish immigration authority (Migri), company registration, licensing, permits, and notifications required for starting a business, especially for foreign entrepreneurs, etc. "
        ),
        "retrieve": (
            "Questions broadly related to business, business planning, business opportunities, startups, entrepreneurship, employment, unemployment, pensions, insurance, social benefits, and similar topics"
            "This includes questions about specific business opportunities (e.g., for specific expertise, area, topic) or suggestions. "
        ),
        "unrelated": (
            "Questions not related to business, entrepreneurship, startups, employment, unemployment, pensions, insurance, social benefits, or similar topics, "
            "or those related to other countries or cities instead of Finland."
        )
    }

    SYS_PROMPT = """Act as a router to select specific tools or functions based on user's question. 
                 - Analyze the given question and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy with the question. 
                   The dictionary has tool names as keys and their descriptions as values. 
                 - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
                 - For questions mentioning any other country except Finland, or any other city except a Finnish city, output 'unrelated'.
                """

    # Define the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", """Here is the question:
                        {question}
                        Here is the tool selection dictionary:
                        {tool_selection}
                        Output the required tool.
                    """),
        ]
    )

    # Pass the inputs to the prompt
    inputs = {
        "question": question,
        "tool_selection": tool_selection
    }

    # Invoke the chain
    tool = (prompt | st.session_state.router_llm |
            StrOutputParser()).invoke(inputs)
    # Remove backslashes and extra spaces
    tool = re.sub(r"[\\'\"`]", "", tool.strip())
    if not "unrelated" in tool:
        print(
            f"Invoking {tool} tool through {st.session_state.router_llm.model_name}")
    if "websearch" in tool:
        print("I need to get recent information from this query.")
    return tool


workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("route_after_grading", route_after_grading)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
workflow.add_node("hybrid_search", hybrid_search)
workflow.add_node("unrelated", handle_unrelated)

# Set conditional entry points
workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve": "retrieve",
        "websearch": "websearch",
        "hybrid_search": "hybrid_search",
        "unrelated": "unrelated"
    },
)

# Add edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"websearch": "websearch", "generate": "generate"},
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("hybrid_search", "generate")
workflow.add_edge("unrelated", "generate")


# Compile app
app = workflow.compile()















# import os
# import streamlit as st
# # Set up environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["USER_AGENT"] = "AgenticRAG/1.0" 
# os.environ["TAVILY_API_KEY"]=st.secrets["TAVILY_API_KEY"]
# os.environ["GROQ_API_KEY"]=st.secrets["GROQ_API_KEY"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# from langchain_community.document_loaders import WebBaseLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
# from langgraph.graph import END, StateGraph
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.documents import Document
# from typing_extensions import TypedDict
# from typing import List
# from PyPDF2 import PdfReader
# from tavily import TavilyClient
# from langchain_groq.chat_models import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# import re
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from sentence_transformers import SentenceTransformer, util
# import spacy
# import warnings
# import logging
# from langchain.chains import RetrievalQA
# import sys
# from langchain.retrievers.document_compressors import FlashrankRerank
# from langchain.retrievers import ContextualCompressionRetriever
# import requests
# from bs4 import BeautifulSoup
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field
# from langchain_openai import OpenAIEmbeddings

# ########################Resolve or suppress warnings
# # Set global logging level to ERROR
# logging.basicConfig(level=logging.ERROR, force=True)
# # Suppress all SageMaker logs
# logging.getLogger("sagemaker").setLevel(logging.CRITICAL)
# logging.getLogger("sagemaker.config").setLevel(logging.CRITICAL)

# # Ignore the specific FutureWarning from Hugging Face Transformers
# warnings.filterwarnings(
#     "ignore", 
#     message="`clean_up_tokenization_spaces` was not set.*",
#     category=FutureWarning
# )
# # General suppression for other warnings (optional)
# warnings.filterwarnings("ignore")
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# ###################################################

# # Define paths and parameters
# data_file_path = 'Becoming an entrepreneur in Finland.md'
# DATA_FOLDER = 'data'
# persist_directory_openai = 'data/chroma_db_llamaparse-openai'
# persist_directory_huggingface = 'data/chroma_db_llamaparse-huggincface'
# collection_name = 'rag'
# CHUNK_SIZE = 3000
# CHUNK_OVERLAP = 200

# def remove_tags(soup):
#     # Remove unwanted tags
#     for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
#         element.decompose()

#     # Extract text while preserving structure
#     content = ""
#     for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
#         text = element.get_text(strip=True)
#         if element.name.startswith('h'):
#             level = int(element.name[1])
#             content += '#' * level + ' ' + text + '\n\n'  # Markdown-style headings
#         elif element.name == 'p':
#             content += text + '\n\n'
#         elif element.name == 'li':
#             content += '- ' + text + '\n'
#     return content

# #@st.cache_data
# def get_info(URLs):
#     """
#     Fetch and return contact information from predefined URLs.
#     """
#     combined_info = ""
#     for url in URLs:
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.text, "html.parser")
#                 combined_info += "URL: " + url + ": " + remove_tags(soup) + "\n\n" 
#             else:
#                 combined_info += f"Failed to retrieve information from {url}\n\n"
#         except Exception as e:
#             combined_info += f"Error fetching URL {url}: {e}\n\n"
#     return combined_info

# #@st.cache_data
# def staticChunker(folder_path):
#     docs = []
#     print(f"Creating chunks. CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")

#     # Loop through all .md files in the folder
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".md"):
#             file_path = os.path.join(folder_path, file_name)
#             print(f"Processing file: {file_path}")

#             # Load documents from the Markdown file
#             loader = UnstructuredMarkdownLoader(file_path)
#             documents = loader.load()

#             # Add file-specific metadata (optional)
#             for doc in documents:
#                 doc.metadata["source_file"] = file_name

#             # Split loaded documents into chunks
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#             chunked_docs = text_splitter.split_documents(documents)
#             docs.extend(chunked_docs)
#     return docs

# #@st.cache_resource
# def load_or_create_vs(persist_directory):
#     # Check if the vector store directory exists
#     if os.path.exists(persist_directory):
#         print("Loading existing vector store...")
#         # Load the existing vector store
#         vectorstore = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=st.session_state.embed_model,
#             collection_name=collection_name
#         )
#     else:
#         print("Vector store not found. Creating a new one...\n")
#         docs = staticChunker(DATA_FOLDER)
#         print("Computing embeddings...")
#         # Create and persist a new Chroma vector store
#         vectorstore = Chroma.from_documents(
#             documents=docs,
#             embedding=st.session_state.embed_model,
#             persist_directory=persist_directory,
#             collection_name=collection_name
#         )
#         print('Vector store created and persisted successfully!')

#     return vectorstore

# def initialize_app(model_name, selected_embedding_model, selected_routing_model, selected_grading_model, hybrid_search, internet_search, answer_style):
#     """
#     Initialize embeddings, vectorstore, retriever, and LLM for the RAG workflow.
#     Reinitialize components only if the selection has changed.
#     """
#     # Track current state to prevent redundant initialization
#     if "current_model_state" not in st.session_state:
#         st.session_state.current_model_state = {
#             "answering_model": None,
#             "embedding_model": None,
#             "routing_model": None,
#             "grading_model": None,
#         }

#     # Check if models or settings have changed
#     state_changed = (
#         st.session_state.current_model_state["answering_model"] != model_name or
#         st.session_state.current_model_state["embedding_model"] != selected_embedding_model or
#         st.session_state.current_model_state["routing_model"] != selected_routing_model or
#         st.session_state.current_model_state["grading_model"] != selected_grading_model
#     )

#     # Reinitialize components only if settings have changed
#     if state_changed:
#         st.session_state.embed_model = initialize_embedding_model(selected_embedding_model)
        
#         # Update vectorstore
#         persist_directory = persist_directory_openai if "text-" in selected_embedding_model else persist_directory_huggingface
#         st.session_state.vectorstore = load_or_create_vs(persist_directory)
#         st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
        
#         st.session_state.llm = initialize_llm(model_name, answer_style)
#         st.session_state.router_llm = initialize_router_llm(selected_routing_model)
#         st.session_state.grader_llm = initialize_grading_llm(selected_grading_model)
#         st.session_state.doc_grader = initialize_grader_chain()

#         # Save updated state
#         st.session_state.current_model_state.update({
#             "answering_model": model_name,
#             "embedding_model": selected_embedding_model,
#             "routing_model": selected_routing_model,
#             "grading_model": selected_grading_model,
#         })

#     print(f"Using LLM: {model_name}, Router LLM: {selected_routing_model}, Grader LLM:{selected_grading_model}, embedding model: {selected_embedding_model}")

#     return workflow.compile()

# #@st.cache_resource
# def initialize_llm(model_name, answer_style):
#     if "llm" not in st.session_state or st.session_state.llm.model_name != model_name:
#         if answer_style == "Concise":
#             temperature = 0.0
#         elif answer_style == "Moderate":
#             temperature = 0.0
#         elif answer_style == "Explanatory":
#             temperature = 0.0

#         if "gpt-" in model_name:
#             st.session_state.llm = ChatOpenAI(model=model_name, temperature=temperature)
#         else:
#             st.session_state.llm = ChatGroq(model=model_name, temperature=temperature)

#     return st.session_state.llm


# def initialize_embedding_model(selected_embedding_model):
#     # Check if the embed_model exists in session_state
#     if "embed_model" not in st.session_state:
#         st.session_state.embed_model = None

#     # Check if the current model matches the selected one
#     current_model_name = None
#     if st.session_state.embed_model:
#         if hasattr(st.session_state.embed_model, "model"):
#             current_model_name = st.session_state.embed_model.model
#         elif hasattr(st.session_state.embed_model, "model_name"):
#             current_model_name = st.session_state.embed_model.model_name

#     # Initialize a new model if it doesn't match the selected one
#     if current_model_name != selected_embedding_model:
#         if "text-" in selected_embedding_model:
#             st.session_state.embed_model = OpenAIEmbeddings(model=selected_embedding_model)
#         else:
#             st.session_state.embed_model = HuggingFaceEmbeddings(model_name=selected_embedding_model)

#     return st.session_state.embed_model

# #@st.cache_resource
# def initialize_router_llm(selected_routing_model):
#     if "router_llm" not in st.session_state or st.session_state.router_llm.model_name != selected_routing_model:
#         if "gpt-" in selected_routing_model:
#             st.session_state.router_llm = ChatOpenAI(model=selected_routing_model, temperature=0.0)
#         else:
#             st.session_state.router_llm = ChatGroq(model=selected_routing_model, temperature=0.0)
    
#     return st.session_state.router_llm

# #@st.cache_resource
# def initialize_grading_llm(selected_grading_model):
#     if "grader_llm" not in st.session_state or st.session_state.grader_llm.model_name != selected_grading_model:
#         if "gpt-" in selected_grading_model:
#             st.session_state.grader_llm = ChatOpenAI(model=selected_grading_model, temperature=0.0, max_tokens = 16000)
#         else:
#             st.session_state.grader_llm = ChatGroq(model=selected_grading_model, temperature=0.0)
    
#     return st.session_state.grader_llm

# model_list = [
#     "llama-3.1-8b-instant",
#     "llama-3.3-70b-versatile",
#     "llama3-70b-8192",   
#     "llama3-8b-8192", 
#     "mixtral-8x7b-32768", 
#     "gemma2-9b-it",
#     "gpt-4o-mini",
#     "gpt-4o",
#     "deepseek-r1-distill-llama-70b"
#     ]

# rag_prompt = PromptTemplate(
#     template = r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#                 You are a helpful, highly accurate and trustworthy assistant specialized in answering questions related to business, entrepreneurship, and the realted matters in Finland. 
#                 Your responses must strictly adhere to the provided context, answer style, and question's language using the follow rules:

#                 1. **Question and answer language**: 
#                 - Detect the question's language (e.g., English, Finnish, Russian, Estonian, Arabic) and answer in the same language. The context could be in English or any other languge. 
#                 - **very important**: make sure that your response is in the same language as the question's. 
#                 2. If the context documents contain 'Internt search results' in 'page_content' field, always consider them in your response. 

#                 3. **Context-Only Answers with a given answer style**:
#                 - Always base your answers on the provided context and answer style.
#                 - If the context does not contain relevant information, respond with: 'No information found. Try rephrasing the query.'
#                 - If the context contains some pieces of the required information, answer with that information and very briefly mention that the answer to other parts could not be found.
#                 - If the context explicitly states 'I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland,' output this context verbatim.

#                 4. **Response style**:
#                 - Address the query directly without unnecessary or speculative information.
#                 - Do not draw from your knowledge base; strictly use the given context. However, take some liberty to provide more explanations and illustrations for better clarity and demonstration from your knowledge and experience only if answer style is "Moderate" or "Explanatory". 
#                 5. **Answer style**
#                 - If answer style = "Concise", generate a concise answer. 
#                 - If answer style = "Moderate", use a moderate approach to generate answer where you can provide a little bit more explanation and elaborate the answer to improve clarity, integrating your own experience. 
#                 - If answer style = "Explanatory", provide a detailed and elaborated answer by providing more explanations with examples and illustrations to improve clarity in best possible way, integrating your own experience. However, the explanations, examples and illustrations should be strictly based on the context. 

#                 6. **Conversational tone**
#                  - Maintain a conversational and helping style which should tend to guide the user and provide him help, hints and offers to further help and information. 
#                  - Use simple language. Explain difficult concepts or terms wherever needed. Present the information in the best readable form.

#                 7. **Formatting Guidelines**:
#                 - Use bullet points for lists.
#                 - Include line breaks between sections for clarity.
#                 - Highlight important numbers, dates, and terms using **bold** formatting.
#                 - Create tables wherever appropriate to present data clearly.
#                 - If there are discrepancies in the context, clearly explain them.

#                 8. **Citation Rules**:
#                 - **very important**: Include citations in the answer at all relevant places if they are present in the context. Under no circumstances ignore them. 
#                 - For responses based on the context documents containing 'page_content' field of 'Smart guide results:', cite the document name and page number with each piece of information in the format: [document_name, page xx].
#                 - For responses based on the documents containing the 'page_content' field of 'Internet search results:', include all the URLs in hyperlink form returned by the websearch. **very important**: The URLs should be labelled with the website name. 
#                 - Do not invent any citation or URL. Only use the citation or URL in the context. 

#                 9. **Hybrid Context Handling**:
#                 - If and only if the context contains documents with two different 'page_content' with the names 'Smart guide results:' and 'Internet search results:', structure your response in corresponding sections with the following headings:
#                     - **Smart guide results**: Include data from 'Smart guide results' and its citations in the format: [document_name, page xx]. If the 'Smart guide results' do not contain any information relevant to the question, output 'No information found' in this section.
#                     - **Internet search results**: Include data from 'Internet search results' and its citations (URLs). 
#                     - Ensure that you create two separate sections if and only if the context contain documents with two different 'page_content': 'Smart guide results:' and 'Internet search results:'.
#                     - Do not create two different sections or mention 'Smart guide results:' or 'Internet search results:' in your response if the context does not contain documents with two different 'page_content': 'Smart guide results:' and 'Internet search results:'.
#                     - Always include the document with 'page_content' of 'Internet search results:' in your response.
#                     - If answer style = "Explanatory", both the sections should be detailed and should contain all the points relevant to the question.
#                 10. **Integrity and Trustworthiness**:
#                 - Ensure every part of your response complies with these rules.

#                 <|eot_id|><|start_header_id|>user<|end_header_id|>
#                 Question: {question} 
#                 Context: {context} 
#                 Answer style: {answer_style}
#                 Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#                 input_variables=["question", "context", "answer_style"]

# )

# def initialize_grader_chain():
#     # Data model for LLM output format
#     class GradeDocuments(BaseModel):
#         """Binary score for relevance check on retrieved documents."""
#         binary_score: str = Field(
#             description="Documents are relevant to the question, 'yes' or 'no'"
#         )

#     # LLM for grading
#     structured_llm_grader = st.session_state.grader_llm.with_structured_output(GradeDocuments)

#     # Prompt template for grading
#     SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.

#     Follow these instructions for grading:
#     - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
#     - Your grade should be either 'Yes' or 'No' to indicate whether the document is relevant to the question or not."""

#     grade_prompt = ChatPromptTemplate.from_messages([
#         ("system", SYS_PROMPT),
#         ("human", """Retrieved document:
#     {documents}
#     User question:
#     {question}
#     """),
#     ])

#     # Build grader chain
#     return grade_prompt | structured_llm_grader

# def grade_documents(state):
#     question = state["question"]
#     documents = state.get("documents", [])
#     filtered_docs = []

#     if not documents:
#         print("No documents retrieved for grading.")
#         return {"documents": [], "question": question, "web_search_needed": "Yes"}

#     print(f"Grading retrieved documents with {st.session_state.grader_llm.model_name}")

#     for count, doc in enumerate(documents):
#         try:
#             # Evaluate document relevance
#             score = st.session_state.doc_grader.invoke({"documents": [doc], "question": question})
#             print(f"Chunk {count} relevance: {score}")
#             if score.binary_score == "Yes":
#                 filtered_docs.append(doc)
#         except Exception as e:
#             print(f"Error grading document chunk {count}: {e}")

#     web_search_needed = "Yes" if not filtered_docs else "No"
#     return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}


# def route_after_grading(state):
#     web_search_needed = state.get("web_search_needed", "No")
#     print(f"Routing decision based on web_search_needed={web_search_needed}")
#     if web_search_needed == "Yes":
#         return "websearch"
#     else:
#         return "generate"

# # Define graph state class
# class GraphState(TypedDict):
#     question: str
#     generation: str
#     web_search_needed: str
#     documents: List[Document]
#     answer_style: str

# def retrieve(state):
#     print("Retrieving documents")
#     question = state["question"]
#     documents = st.session_state.retriever.invoke(question)
#     return {"documents": documents, "question": question}

# def format_documents(documents):
#     """Format documents into a single string for context."""
#     return "\n\n".join(doc.page_content for doc in documents)

# def generate(state):
#     question = state["question"]
#     documents = state.get("documents", [])
#     answer_style = state.get("answer_style", "Concise")

#     if "llm" not in st.session_state:
#         st.session_state.llm = initialize_llm(st.session_state.selected_model, answer_style)

#     rag_chain = rag_prompt | st.session_state.llm | StrOutputParser()

#     if not documents:
#         print("No documents available for generation.")
#         return {"generation": "No relevant documents found.", "documents": documents, "question": question}

#     tried_models = set()
#     original_model = st.session_state.selected_model
#     current_model = original_model

#     while len(tried_models) < len(model_list):
#         try:
#             tried_models.add(current_model)
#             st.session_state.llm = initialize_llm(current_model, answer_style)
#             rag_chain = rag_prompt | st.session_state.llm | StrOutputParser()

#             #context = format_documents(documents)
#             context = documents
#             generation = rag_chain.invoke({"context": context, "question": question, "answer_style": answer_style})

#             print(f"Generating a {answer_style} length response.")
#             #print(f"Response generated with {st.session_state.llm.model_name} model.")
#             print("Done.")

#             if current_model != original_model:
#                 print(f"Reverting to original model: {original_model}")
#                 st.session_state.llm = initialize_llm(original_model, answer_style)

#             return {"documents": documents, "question": question, "generation": generation}

#         except Exception as e:
#             error_message = str(e)
#             if "rate_limit_exceeded" in error_message or "Request too large" in error_message or "Please reduce the length of the messages or completion" in error_message:
#                 print(f"Model's rate limit exceeded or request too large.")
#                 current_model = model_list[(model_list.index(current_model) + 1) % len(model_list)]
#                 print(f"Switching to model: {current_model}")
#             else:
#                 return {
#                     "generation": f"Error during generation: {error_message}",
#                     "documents": documents,
#                     "question": question,
#                 }

#     return {
#         "generation": "Unable to process the request due to limitations across all models.",
#         "documents": documents,
#         "question": question,
#     }

# def handle_unrelated(state):
#     question = state["question"]
#     documents = state.get("documents",[])
#     response = "I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland. Could you please rephrase your question to focus on these topics?"
#     documents.append(Document(page_content=response))
#     return {"generation": response, "documents": documents, "question": question}

# def hybrid_search(state):
#     question = state["question"]
#     print("Invoking retriever...")
#     vector_docs = st.session_state.retriever.invoke(question)
#     web_docs = web_search({"question": question})["documents"]
    
#     # Add headings to distinguish between vector and web search results
#     vector_results = [Document(page_content="Smart guide results:" + doc.page_content) for doc in vector_docs]

#     # Check if any web_docs already contain "Internet search results:"
#     web_results_contain_header = any("Internet search results:" in doc.page_content for doc in web_docs)

#     # Add "Internet search results:" only if not already present in any web doc
#     if not web_results_contain_header:
#         web_results = [
#             Document(page_content="Internet search results:" + doc.page_content) for doc in web_docs
#         ]
#     else:
#         web_results = web_docs  # Keep web_docs unchanged if they already contain the header

    
#     combined_docs = vector_results + web_results
#     return {"documents": combined_docs, "question": question}

# def web_search(state):
#     if "tavily_client" not in st.session_state:
#         st.session_state.tavily_client = TavilyClient()
#     question = state["question"]
#     question = re.sub(r'\b\w+\\|Internet search\b', '', question).strip()
#     question = question + " in Finland"
#     documents = state.get("documents", [])
#     try:
#         print("Invoking internet search...")
#         search_result = st.session_state.tavily_client.get_search_context(
#             query=question,
#             search_depth="basic", #can be switched to 'advanced' mode that requires 2 credits per search.
#             max_tokens=4000,
#             max_results = 10,
#             include_domains = [
#                 "migri.fi",
#                 "enterfinland.fi",
#                 "businessfinland.fi",
#                 "kela.fi",
#                 "vero.fi",
#                 "suomi.fi",
#                 "valvira.fi",
#                 "finlex.fi",
#                 "hus.fi",
#                 "lvm.fi",
#                 "thefinlandbusinesspress.fi",
#                 "infofinland.fi",
#                 "ely-keskus.fi",
#                 "yritystulkki.fi",
#                 "tem.fi",
#                 "prh.fi"
#                 ],
#         )
#         # Handle different types of results
#         if isinstance(search_result, str):
#             web_results = search_result
#         elif isinstance(search_result, dict) and "documents" in search_result:
#             web_results = "Internet search results:".join([doc.get("content", "") for doc in search_result["documents"]])
#         else:
#             web_results = "No valid results returned by TavilyClient."
#         web_results_doc = Document(page_content=web_results)
#         documents.append(web_results_doc)
#     except Exception as e:
#         print(f"Error during web search: {e}")
#         # Ensure workflow can continue gracefully
#         documents.append(Document(page_content=f"Web search failed: {e}"))
#     return {"documents": documents, "question": question}


# # # Router function
# def route_question(state):
#     question = state["question"]
#     hybrid_search_enabled = state.get("hybrid_search", False)
#     internet_search_enabled = state.get("internet_search", False)
    
#     if hybrid_search_enabled:
#         return "hybrid_search"
    
#     if internet_search_enabled:
#         return "websearch"

#     tool_selection = {
#     "websearch": (
#         "Questions requiring current statistics or real-time information such as tax rate, taxation rules, taxable incomes, tax exemptions, the tax filing process, immigration or visa process or questions related to Finnish immigration authority (Migri), company registration, licensing, permits, and notifications required for starting a business, especially for foreign entrepreneurs, etc. "
#     ),
#     "retrieve": (
#         "Questions broadly related to business, business planning, business opportunities, startups, entrepreneurship, employment, unemployment, pensions, insurance, social benefits, and similar topics"
#         "This includes questions about specific business opportunities (e.g., for specific expertise, area, topic) or suggestions. "
#     ),
#     "unrelated": (
#         "Questions not related to business, entrepreneurship, startups, employment, unemployment, pensions, insurance, social benefits, or similar topics, "
#         "or those related to other countries or cities instead of Finland."
#     )
# }


#     SYS_PROMPT = """Act as a router to select specific tools or functions based on user's question. 
#                  - Analyze the given question and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy with the question. 
#                    The dictionary has tool names as keys and their descriptions as values. 
#                  - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
#                  - For questions mentioning any other country except Finland, or any other city except a Finnish city, output 'unrelated'.
#                 """

#     # Define the ChatPromptTemplate
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", SYS_PROMPT),
#             ("human", """Here is the question:
#                         {question}
#                         Here is the tool selection dictionary:
#                         {tool_selection}
#                         Output the required tool.
#                     """),
#         ]
#     )

#     # Pass the inputs to the prompt
#     inputs = {
#         "question": question,
#         "tool_selection": tool_selection
#     }

#     # Invoke the chain
#     tool = (prompt | st.session_state.router_llm | StrOutputParser()).invoke(inputs)
#     tool = re.sub(r"[\\'\"`]", "", tool.strip()) # Remove backslashes and extra spaces
#     if not "unrelated" in tool:
#         print(f"Invoking {tool} tool through {st.session_state.router_llm.model_name}")
#     if "websearch" in tool:
#         print("I need to get recent information from this query.")
#     return tool

    
# workflow = StateGraph(GraphState)

# # Add nodes
# workflow.add_node("retrieve", retrieve)
# workflow.add_node("grade_documents", grade_documents)
# workflow.add_node("route_after_grading", route_after_grading)
# workflow.add_node("websearch", web_search)
# workflow.add_node("generate", generate)
# workflow.add_node("hybrid_search", hybrid_search)
# workflow.add_node("unrelated", handle_unrelated)

# # Set conditional entry points
# workflow.set_conditional_entry_point(
#     route_question,
#     {
#         "retrieve": "retrieve",
#         "websearch": "websearch",
#         "hybrid_search": "hybrid_search",
#         "unrelated": "unrelated"
#     },
# )

# # Add edges
# workflow.add_edge("retrieve", "grade_documents")
# workflow.add_conditional_edges(
#     "grade_documents",
#     route_after_grading,
#     {"websearch": "websearch", "generate": "generate"},
# )
# workflow.add_edge("websearch", "generate")
# workflow.add_edge("hybrid_search", "generate")
# workflow.add_edge("unrelated", "generate")


# # Compile app
# app = workflow.compile()
