#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import TypedDict, List 
from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph , START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 


# In[2]:


load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY is not set")

# # optional: don’t set the key in code if you have it in .env
# # os.environ["OPENAI_API_KEY"] = api_key
# print(api_key)


google_api_key = "XXXXXXXXXXXXXXXXXXXXXXXXX"

# os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("GOOGLE_API_KEY is not set")



# In[ ]:


class AgentState(TypedDict): 
    messages : List[HumanMessage]
#llm = ChatOpenAI(model ="gpt-4o")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = google_api_key
)

def process(state : AgentState) -> AgentState : 
    response = llm.invoke(state["messages"])
    print(f"\n AI : {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process" ,  process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()


while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "quit":
        break

    if not user_input.strip():
        print("Message vide")
        continue

    agent.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ]
    })


# In[ ]:


class AgentState(TypedDict): 
    messages : List[HumanMessage]
#llm = ChatOpenAI(model ="gpt-4o")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = google_api_key
)

def process(state : AgentState) -> AgentState : 
    response = llm.invoke(state["messages"])
    print(f"\n AI : {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process" ,  process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()


while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "quit":
        break

    if not user_input.strip():
        print("Message vide")
        continue

    agent.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ]
    })

