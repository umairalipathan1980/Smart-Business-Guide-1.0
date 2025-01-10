
import os
import streamlit as st
# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["USER_AGENT"] = "AgenticRAG/1.0" 
os.environ["TAVILY_API_KEY"]=st.secrets["TAVILY_API_KEY"]
os.environ["GROQ_API_KEY"]=st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from typing_extensions import TypedDict
from typing import List
from PyPDF2 import PdfReader
from tavily import TavilyClient
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from sentence_transformers import SentenceTransformer, util
import spacy
import warnings
import logging
from langchain.chains import RetrievalQA
import sys
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings

########################Resolve or suppress warnings
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

#@st.cache_data
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
                combined_info += "URL: " + url + ": " + remove_tags(soup) + "\n\n" 
            else:
                combined_info += f"Failed to retrieve information from {url}\n\n"
        except Exception as e:
            combined_info += f"Error fetching URL {url}: {e}\n\n"
    return combined_info

#@st.cache_data
def staticChunker(folder_path):
    docs = []
    print(f"Creating chunks. CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunked_docs = text_splitter.split_documents(documents)
            docs.extend(chunked_docs)
    return docs

#@st.cache_resource
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
        st.session_state.embed_model = initialize_embedding_model(selected_embedding_model)
        
        # Update vectorstore
        persist_directory = persist_directory_openai if "text-" in selected_embedding_model else persist_directory_huggingface
        st.session_state.vectorstore = load_or_create_vs(persist_directory)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        st.session_state.llm = initialize_llm(model_name, answer_style)
        st.session_state.router_llm = initialize_router_llm(selected_routing_model)
        st.session_state.grader_llm = initialize_grading_llm(selected_grading_model)
        st.session_state.doc_grader = initialize_grader_chain()

        # Save updated state
        st.session_state.current_model_state.update({
            "answering_model": model_name,
            "embedding_model": selected_embedding_model,
            "routing_model": selected_routing_model,
            "grading_model": selected_grading_model,
        })

    print(f"Using LLM: {model_name}, Router LLM: {selected_routing_model}, Grader LLM:{selected_grading_model}, embedding model: {selected_embedding_model}")

    return workflow.compile()

#@st.cache_resource
def initialize_llm(model_name, answer_style):
    if "llm" not in st.session_state or st.session_state.llm.model_name != model_name:
        if answer_style == "Concise":
            temperature = 0.0
        elif answer_style == "Moderate":
            temperature = 0.2
        elif answer_style == "Explanatory":
            temperature = 0.4

        if "gpt-" in model_name:
            st.session_state.llm = ChatOpenAI(model=model_name, temperature=temperature)
        else:
            st.session_state.llm = ChatGroq(model=model_name, temperature=temperature)

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
            st.session_state.embed_model = OpenAIEmbeddings(model=selected_embedding_model)
        else:
            st.session_state.embed_model = HuggingFaceEmbeddings(model_name=selected_embedding_model)

    return st.session_state.embed_model



#@st.cache_resource
def initialize_router_llm(selected_routing_model):
    if "router_llm" not in st.session_state or st.session_state.router_llm.model_name != selected_routing_model:
        if "gpt-" in selected_routing_model:
            st.session_state.router_llm = ChatOpenAI(model=selected_routing_model, temperature=0.0)
        else:
            st.session_state.router_llm = ChatGroq(model=selected_routing_model, temperature=0.0)
    
    return st.session_state.router_llm

#@st.cache_resource
def initialize_grading_llm(selected_grading_model):
    if "grader_llm" not in st.session_state or st.session_state.grader_llm.model_name != selected_grading_model:
        if "gpt-" in selected_grading_model:
            st.session_state.grader_llm = ChatOpenAI(model=selected_grading_model, temperature=0.0)
        else:
            st.session_state.grader_llm = ChatGroq(model=selected_grading_model, temperature=0.0)
    
    return st.session_state.grader_llm


model_list = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",   
    "llama3-8b-8192", 
    "mixtral-8x7b-32768", 
    "gemma2-9b-it",
    "gpt-4o-mini",
    "gpt-4o"
    ]

rag_prompt = PromptTemplate(
    template = r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a highly accurate and trustworthy assistant specialized in answering questions related to business and entrepreneurship in Finland. 
                Your responses must strictly adhere to the provided context, answer style, using the follow these rules:

                1. **Context-Only Answers with a given answer style**:
                - Always base your answers on the provided context and answer style.
                - If the context does not contain relevant information, respond with: 'No information found.'
                - If the context explicitly states 'I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland,' output this context verbatim.

                2. **Response style**:
                - Address the query directly without unnecessary or speculative information.
                - Do not draw from your knowledge base; strictly use the given context. However, take some liberty to provide more explanations and illustrations for better clarity and demonstration from your knowledge and experience only if answer style is "Moderate" or "Explanatory". 
                - Consider the given answer style to produce the answer. If answer style = "Concise", generate a precise and concise answer. If answer style = "Moderate", use a moderate approach to generate answer
                  where you can provide a little bit more explanation and elaborate the answer to improve clarity, integrating your own experience. If answer style = "Explanatory", elaborate the answer to provide more explanations with examples and illustrations to improve clarity in best possible way, integrating your own experience.
                  However, the explanations, examples and illustrations should be strictly based on the context. 

                3. **Conversational tone**
                 - Maintain a conversational but professional tone. 
                 - Use simple language. Explain difficult concepts or terms wherever needed.

                4. **Formatting Guidelines**:
                - Use bullet points for lists.
                - Include line breaks between sections for clarity.
                - Highlight important numbers, dates, and terms using **bold** formatting.
                - Create tables wherever appropriate to present data clearly.
                - If there are discrepancies in the context, clearly explain them.

                5. **Citation Rules**:
                - For responses based on vectorstore retrieval, cite the document name and page number with each piece of information in the format: [document_name, page xx].
                - For the answer compiled from the context from multiple documents, use the format: document_name 1 [page xx, yy, zz, ...], document_name 2 [page xx, yy, zz, ...].
                - For responses derived from websearch results, include all the URLs returned by the websearch, each on a new line.
                - Do not invent any citation or URL. Only use the citation or URL in the context. 

                6. **Hybrid Context Handling**:
                - If the context contains two different sections with the names 'Smart guide results:' and 'Internet search results:', structure your response in corresponding sections with the following headings:
                    - **Smart guide results**: Include data from vectorstore retrieval and its citations in the format: [document_name, page xx].
                    - **Internet search results**: Include data from websearch and its citations (URLs). This does not mean only internet URLs, but all the data in 'Internet search results:' along with URLs.
                    - Do not combine the data in the two sections. Create two separate sections. 

                7. **Integrity and Trustworthiness**:
                - Never provide information that is not explicitly found in the context, unless the answer style is "explanatory" in which you can add some examples and explanations only for illustration purposes.
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
    structured_llm_grader = st.session_state.grader_llm.with_structured_output(GradeDocuments)

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

    print(f"Grading retrieved documents with {st.session_state.grader_llm.model_name}")

    for count, doc in enumerate(documents):
        try:
            # Evaluate document relevance
            score = st.session_state.doc_grader.invoke({"documents": [doc], "question": question})
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
        st.session_state.llm = initialize_llm(st.session_state.selected_model, answer_style)

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

            context = format_documents(documents)
            generation = rag_chain.invoke({"context": context, "question": question, "answer_style": answer_style})

            print(f"Generating a {answer_style} length response.")
            print(f"Response generated with {st.session_state.llm.model_name} model.")
            print("Done.")

            if current_model != original_model:
                print(f"Reverting to original model: {original_model}")
                st.session_state.llm = initialize_llm(original_model, answer_style)

            return {"documents": documents, "question": question, "generation": generation}

        except Exception as e:
            error_message = str(e)
            if "rate_limit_exceeded" in error_message or "Request too large" in error_message or "Please reduce the length of the messages or completion" in error_message:
                print(f"Model's rate limit exceeded or request too large.")
                current_model = model_list[(model_list.index(current_model) + 1) % len(model_list)]
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
    documents = state.get("documents",[])
    response = "I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland. Could you please rephrase your question to focus on these topics?"
    documents.append(Document(page_content=response))
    return {"generation": response, "documents": documents, "question": question}

def hybrid_search(state):
    question = state["question"]
    print("Invoking retriever...")
    vector_docs = retriever.invoke(question)
    web_docs = web_search({"question": question})["documents"]
    
    # Add headings to distinguish between vector and web search results
    vector_results = [Document(page_content="Smart guide results:\n\n" + doc.page_content) for doc in vector_docs]
    web_results = [Document(page_content="\n\nInternet search results:\n\n" + doc.page_content) for doc in web_docs]
    
    combined_docs = vector_results + web_results
    return {"documents": combined_docs, "question": question}

#@st.cache_data
def web_search(state):
    if "tavily_client" not in st.session_state:
        st.session_state.tavily_client = TavilyClient()
    question = state["question"]
    question = re.sub(r'\b\w+\\|Internet search\b', '', question).strip()
    question = question + " in Finland"
    documents = state.get("documents", [])
    try:
        print("Invoking internet search...")
        search_result = tavily_client.get_search_context(
            query=question,
            search_depth="advanced",
            max_tokens=4000
        )
        # Handle different types of results
        if isinstance(search_result, str):
            web_results = search_result
        elif isinstance(search_result, dict) and "documents" in search_result:
            web_results = "\n".join([doc.get("content", "") for doc in search_result["documents"]])
        else:
            web_results = "No valid results returned by TavilyClient."
        web_results_doc = Document(page_content=web_results)
        documents.append(web_results_doc)
    except Exception as e:
        print(f"Error during web search: {e}")
        # Ensure workflow can continue gracefully
        documents.append(Document(page_content=f"Web search failed: {e}"))
    return {"documents": documents, "question": question}

def get_contact_tool(state):
    """
    Execute the 'get_contact_info' tool to fetch information.
    """
    contact_urls = [
        'https://migri.fi/en/contact-information',
        'https://migri.fi/en/communications'
    ]
    question = state["question"]
    documents = state.get("documents", [])
    try:
        contact_info = get_info(contact_urls)
        web_results_doc = Document(page_content=contact_info)
        documents.append(web_results_doc)
        return {
            "generation": contact_info,
            "documents": documents,
            "question": question
        }
    except Exception as e:
        return {
            "generation": f"Error fetching contact information: {e}",
            "documents": [],
            "question": question
        }

def get_tax_info(state):
    """
    Execute the 'get_contact_info' tool to fetch information.
    """
    tax_rates_url = [
        'https://www.vero.fi/en/businesses-and-corporations/taxes-and-charges/vat/rates-of-vat/',
        'https://www.expat-finland.com/living_in_finland/tax.html?utm_source=chatgpt.com',
        'https://finlandexpat.com/tax-in-finland/?utm_source=chatgpt.com'
    ]
    question = state["question"]
    documents = state.get("documents", [])
    try:
        tax_info = get_info(tax_rates_url)
        web_results_doc = Document(page_content=tax_info)
        documents.append(web_results_doc)
        return {
            "generation": tax_info,
            "documents": documents,
            "question": question
        }
    except Exception as e:
        return {
            "generation": f"Error fetching contact information: {e}",
            "documents": [],
            "question": question
        }

def get_registration_info(state):
    """
    Execute the 'get_contact_info' tool to fetch information.
    """
    reg_links = ['https://www.suomi.fi/company/starting-a-business/forms-of-enterprise/guide/choosing-the-form-of-business/company-registration']
    question = state["question"]
    documents = state.get("documents", [])
    try:
        reg_info = get_info(reg_links)
        web_results_doc = Document(page_content=reg_info)
        documents.append(web_results_doc)
        return {
            "generation": reg_info,
            "documents": documents,
            "question": question
        }
    except Exception as e:
        return {
            "generation": f"Error fetching contact information: {e}",
            "documents": [],
            "question": question
        }

def get_licensing_info(state):
    """
    Execute the 'get_contact_info' tool to fetch information.
    """
    licenses = [
    'https://www.suomi.fi/company/starting-a-business/foreign-entrepreneurs/guide/licenses-and-notifications-required-of-foreign-entrepreneurs',
    'https://yritystulkki.fi/fi/alue/oulu/english/industries_requiring_license/'
    ]
    question = state["question"]
    documents = state.get("documents", [])
    try:
        licenses_info = get_info(licenses)
        web_results_doc = Document(page_content=licenses_info)
        documents.append(web_results_doc)
        return {
            "generation": licenses_info,
            "documents": documents,
            "question": question
        }
    except Exception as e:
        return {
            "generation": f"Error fetching contact information: {e}",
            "documents": [],
            "question": question
        }

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
        "get_tax_info": "question related to tax related information including current tax rates, taxation rules, taxable incomes, tax exemption, tax filing process, etc., but not asking information about any other country or city except Finland.",
        "get_contact_tool": "question related to contact information of the Finnish Immigration Service, also known as Migri (but not asking information about any other country or city except Finland.)",
        "get_registration_info": "question specifically related to the process of company registration. This does not include questions related to starting a business. The question should not ask information about any other country or city except Finland.",
        "get_licensing_info": "question related to licensing, permits and notifications required for foreign entrepreneurs to start a business. This does not include questions related to residence permits. The question should not ask information about any other country or city except Finland.",
        "websearch": "questions related to residence permit, visa, and moving to Finland or the questions requiring current statistics, but not asking information about any other country or city except Finland.",
        "retrieve": "All other question related to business, entrepreneurship, job, unemployment, pension, insurance, social benefits, etc not covered by the other tools, but not asking information about any other country or city except Finland.)",
        "unrelated": "Questions not related to business, entrepreneurship, job, unemployment, pension, insurance, social benefits, etc in Finland, or related to other countries instead of Finland."
    }

    SYS_PROMPT = """Act as a router to select specific tools or functions based on user's question. 
                 - Analyze the given question and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy with the question. 
                   The dictionary has tool names as keys and their descriptions as values. 
                 - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
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
    tool = (prompt | st.session_state.router_llm | StrOutputParser()).invoke(inputs)
    tool = re.sub(r"[\\'\"`]", "", tool.strip()) # Remove backslashes and extra spaces
    if not "unrelated" in tool:
        print(f"Invoking {tool} tool through {st.session_state.router_llm.model_name}")
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
workflow.add_node("get_contact_tool", get_contact_tool)
workflow.add_node("get_tax_info", get_tax_info)
workflow.add_node("get_registration_info", get_registration_info)
workflow.add_node("get_licensing_info", get_licensing_info)
workflow.add_node("hybrid_search", hybrid_search)
workflow.add_node("unrelated", handle_unrelated)

# Set conditional entry points
workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve": "retrieve",
        "websearch": "websearch",
        "get_contact_tool": "get_contact_tool",
        "get_tax_info": "get_tax_info",
        "get_registration_info": "get_registration_info",
        "get_licensing_info": "get_licensing_info",
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
workflow.add_edge("get_contact_tool", "generate")
workflow.add_edge("get_tax_info", "generate")
workflow.add_edge("get_registration_info", "generate")
workflow.add_edge("get_licensing_info", "generate")
workflow.add_edge("hybrid_search", "generate")
workflow.add_edge("unrelated", "generate")


# Compile app
app = workflow.compile()



