#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import TypedDict, List , Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph , START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 


load_dotenv()


# In[2]:


class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]
    


# In[ ]:


google_api_key = "AIzaSyBI0hFu34XX609bRT4z3ulX0nxEZ5UsAaA"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = google_api_key
)


def process(state : AgentState) -> AgentState : 
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\n AI : {response.content}")
    print("Current State : ",   state["messages"])

    return state


graph = StateGraph(AgentState)
graph.add_node("process" ,  process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("\nYou : ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content = user_input))

    result = agent.invoke({"messages": conversation_history})

    conversation_history = result["messages"]
    user_input = input("\nYou : ")



with open("logging.txt", "w") as file : 
    file.write("Your conversation Log :\n") 

    for message in conversation_history: 
        if isinstance(message, HumanMessage): 
            file.write(f"You :{message.content}\n")
        elif isinstance(message, AIMessage): 
            file.write(f"AI {message.content}\n \n")
    file.write("End of conversation \n")

print("Converstation saved to logging.txt")
                       


# In[ ]:




