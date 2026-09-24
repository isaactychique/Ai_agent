#!/usr/bin/env python
# coding: utf-8

# In[9]:


from typing import List, Dict, Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages  import BaseMessage , HumanMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the 
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode 

load_dotenv()


# In[10]:


email = Annotated[str, "This hass to be a valid email format!"]
print(email.__metadata__)


# In[11]:


state = {"messages" : ["Hi!"]}
update = {"messages" : ["Nice to meet you!"]}
new_state = {"messages" : ["Nice to meet you!"]}


state = {"messages" : ["Hi!"]}
update = {"messages" : ["Nice to meet you!"]}
new_state = {"messages" : ["Hi" , "Nice to meet you!"]}


# In[ ]:


class AgentState(TypedDict): 
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int , b: int):
    """this is an addition function that adds 2 number together"""
    return a + b

tools = [add]

google_api_key = "AIzaSyBI0hFu34XX609bRT4z3ulX0nxEZ5UsAaA"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key = google_api_key).bind_tools(tools)


def model_call(state : AgentState) -> AgentState : 
    system_prompt = SystemMessage(content = 
                    "You are my AI assistant, please answer my query to the best of your ability.")
    response = llm.invoke([system_prompt] + state["messages"]
                          )
    return {"messages" :[response]}
    

def should_continue(state : AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else : 
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent", 
    should_continue, 
    {
        "continue" : "tools", 
        "end" : END,
    }
)

graph.add_edge("tools" , "our_agent")

app = graph.compile()

def print_stream(stream): 
    for s in stream : 
        message = s["messages"][-1]
        if isinstance(message, tuple) :
            print(message)
        else :
            message.pretty_print()


#inputs = {"messages" : [("user", "add 40 + 12")] }
inputs = {"messages" : [HumanMessage(content ="add 40 + 12 Add 5 +6")] }
print_stream(app.stream(inputs, stream_mode="values"))


# In[ ]:




