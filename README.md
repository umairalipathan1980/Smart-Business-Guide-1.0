# Smart Business Guide 1.0
**Smart guide for entrepreneurship and business planning in Finland based on agentic RAG (Langchain and LangGraph).**   

This Smart Guide assists users with business and entrepreneurship queries in Finland by retrieving information from authetnic guides, performing internet or hybrid searches, and generating tailored responses using agentic AI workflows.  

* Retrieval from authentic business and entrepreneurship guides, parsed through LlamaParse
* Dynamic decisions to select vector database, web search, or a specific tool to extract information from trusted sources.
* A retrieval grading mechanism to assess the relevancy of the retrieved documents and deciding whether to proceed with the retrieved documents and invoking web search.
* Option to select three different answer styles: concise, moderate, and explanatory.
* Options to enforce internet or hybrid search. 

The smart guide offers the options to selection several open-source models as well as OpenAI models. The open-source models are accessed using Groq Cloud API which offer a lightening fast response to queries. Different LLMs can be selected for different nodes in the agentic workflow. 

# How to Use the Code.
**Clone the repository:**
   ```
   git clone https://github.com/umairalipathan1980/Smart-Business-Guide-1.0.git
   cd Smart-Business-Guide-1.0
   ```
**Install the required libraries:**
   ```
   pip install -r requirements.txt
   ```
**Create a folder ".streamlit" in the root directory and create a "secrets.toml" file in it. Set your API keys there as follows:**
   ```
   GROQ_API_KEY = "your_GROQ_api_key" # get a free Groq API key from Groq Cloud
   TAVILY_API_KEY = "your_Tavily_api_key" # get a free Tavily API key
   LANGCHAIN_API_KEY = "your_LangChain_API_key" # get a free API key from LangSmith
   OPENAI_API_KEY = "your_OpenAI_API_KEY" # (optional) for OpenAI models
   ```
