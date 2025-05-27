# triage_ai_assistant/agents/nursebot.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langchain_openai import OpenAI

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# Tool for note-taking
@tool
def take_note(text: str) -> str:
    """Add this symptom description to your notes if it seems medically relevant."""
    return text


llm_with_tools = llm.bind_tools([take_note])

class SymptomState(TypedDict):
    messages: Annotated[list, add_messages]  # Stores conversation history
    notes: list[str]  # Collected notes from patient
    finished: bool

NURSEBOT_SYSINT = (
    "system",
    "You are NurseBot, a friendly and professional virtual clinical assistant. You are designed to talk to patients "
    "and help gather information about their symptoms. Ask clear, conversational questions to understand what the patient is experiencing.\n\n"
    "Do not overwhelm the patient — ask one question at a time. Avoid more than 10 questions. If you feel you've collected enough information, stop asking.. Be kind, and don't attempt to diagnose.\n\n"
    "Once done, don't forget to record symptoms by calling take_note('symptom description'). \n\n"
)

WELCOME_MSG = "Welcome to the MedMacs Hospital. Type `q` to quit. How may I help you today?"

def human_node(state: SymptomState) -> SymptomState:
    last_message = state["messages"][-1]
    print(f"Assistant: {last_message.content}")
    user_input = input("User: ").strip()

    if user_input.lower() in {"q", "quit", "exit", "thank you"}:
        return {**state, "finished": True}

    return {
        **state,
        "messages": state["messages"] + [("user", user_input)]
    }

def chatbot_node(state: SymptomState) -> SymptomState:
    if state["messages"]:
        response = llm_with_tools.invoke([NURSEBOT_SYSINT] + state["messages"])
    else:
        response = AIMessage(content=WELCOME_MSG)

    new_notes = state.get("notes", [])
    finished = state.get("finished", False)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            if call["name"] == "take_note" and "text" in call["args"]:
                new_notes.append(call["args"]["text"])
                finished = True
            

    return {
        **state,
        "messages": state["messages"] + [response],
        "notes": new_notes,
        "finished": finished
    }

def handle_chat(messages: list[str]) -> str:
    history = [("system", NURSEBOT_SYSINT)] + [("user", msg) for msg in messages]
    response = llm_with_tools.invoke(history)
    return response.content

graph_builder = StateGraph(SymptomState)
graph_builder.add_node("human", human_node)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", "human")
graph_builder.add_edge("human", "chatbot")


# def maybe_exit_human_node(state: SymptomState):
#     return END if state.get("finished", False) else "chatbot"

def maybe_exit_chatbot_node(state: SymptomState):
    return END if state.get("finished", False) else "human"


# graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_conditional_edges("chatbot", maybe_exit_chatbot_node)

chat_with_human_graph = graph_builder.compile()

def run_chat(config = {"recursion_limit": 100}):
    state = chat_with_human_graph.invoke({"messages": [], "notes": [], "finished": False}, config)
    return state

if __name__ == "__main__":
    run_chat()
