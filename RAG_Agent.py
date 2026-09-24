#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import List, Dict, Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages  import BaseMessage , HumanMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the 
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode 
from langchain_community.vectorstores import Chroma
import os 

load_dotenv()
GOOGLE_API_KEY="AIzaSyBI0hFu34XX609bRT4z3ulX0nxEZ5UsAaA"
#google_api_key = os.getenv(GOOGLE_API_KEY)
google_api_key = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
     model="gemini-2.5-flash", 
    google_api_key = google_api_key, 
    temperature = 0)  # It's to minimize hallucination. = 0 make the model more deterministic 

# Our embedding Model - has to also be compatible with the LLM

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="embedding-001", 
#     google_api_key=google_api_key
# )



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pdf_path="Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try : 
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")

except Exception as e : 
    print(f"Error loading PDF : {e}")
    raise 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


pages_split = text_splitter.split_documents(pages)  # We now apply this to our pages 
persist_directory = r"/home/dakiotychique17/Documents/Ai_agent/"
collection_name = "stock_market"


if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector stor!")
   
except Exception as e:
    print(f"Created setting up ChromaDB: {str(e)}")
    raise


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


@tool 
def retriever_tool(query : str) -> str : 
    """
    This tool searches and returns the information from the Stock market Performance 2024 document
    """

    docs = retriever.invoke(query)

    if not docs : 
        return "I found no relevant information in the Stock Market Performance 2024 document"
    
    results = []
    for i, doc in enumerate(docs): 
        results.append(f"Document {i+1}: \n {doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict): 
    messages : Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state : AgentState) : 
    """ Check if the last message contains tool calls"""
    result = state["messages"][-1]
    if hasattr(result, 'tool_calls') and len(result.tool_calls) > 0 : 
        return "continue"
    return "end"


system_prompt = """
    You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on document loaded into your knowledge base 
    Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed 
    If you need to llok up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers. 
    """

tools_dict = {our_tool.name : our_tool for our_tool in tools}

def call_llm(state : AgentState) -> AgentState: 
    """Function to call the LLM with the current state. """ 
    messages = list(state["messages"])
    messages = [SystemMessage(content = system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages' : [message]}



def take_action(state : AgentState) -> AgentState: 
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls : 
        print(f"Calling Tool : {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict : 
            print(f"\n Tool: {t['name']} doesn't exist")
            result = "Incorrect tool name, Please retry  and select tool from list of available tools"
        
        else : 
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'] , name = t['name'], content = str(result)))
    
    print("Tools Execution Complete. Back to the model!")
    return {'messages' : results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm", 
    should_continue, 
    {
        "continue" : "retriever_agent", 
        "end" : END 
       
    }
)

graph.add_edge("retriever_agent", "llm")
rag_agent = graph.compile()




def run_rag_agent(): 
    print ("\n ====== RAG AGENT ======")

    while True: 
        user_input = input("\n What is your question: ")
        if user_input.lower() in ['exit', 'quit'] : 
            break 

        messages = [HumanMessage(content= user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n ==== ANSWER ====")
        print(result['messages'][-1].content)


if __name__ == "__main__" : 
    run_rag_agent()


# In[ ]:


#!pip install langchain langchain-community langchain-google-genai
#!pip install langchain_text_splitters
#!pip install pypdf


# In[ ]:


#!pip install -U google-genai langchain-google-genai
from google import genai

client = genai.Client(api_key="AIzaSyBI0hFu34XX609bRT4z3ulX0nxEZ5UsAaA")

models = client.models.list()

for m in models:
    print(m.name)


# In[ ]:





# In[ ]:




