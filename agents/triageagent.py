# triage_ai_assistant/agents/triage_engine.py
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Use Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

from langchain_openai import OpenAI

llm = OpenAI()


# Define prompt templates
nurse_prompt = ChatPromptTemplate.from_template(
    "Role: ER Triage Nurse\nInstructions: Determine ESI (1-5) with reasoning...\nPatient Note: {note}\nPrevious Doctor Input: {doctor_msg}\nYour Response:"
)

doctor_prompt = ChatPromptTemplate.from_template(
    "Role: ER Doctor\nInstructions: List possible diagnoses and comment on priority...\nPatient Note: {note}\nNurse's Assessment: {nurse_msg}\nYour Response:"
)

def nurse_step(state: Dict[str, Any]) -> Dict[str, Any]:
    inputs = {
        "note": state["note"],
        "doctor_msg": state.get("doctor_msg", "")
    }
    response = (nurse_prompt | llm).invoke(inputs)
    if isinstance(response, str):
        return {**state, "nurse_msg": response}
    
    return {**state, "nurse_msg": response.content}

def doctor_step(state: Dict[str, Any]) -> Dict[str, Any]:
    inputs = {
        "note": state["note"],
        "nurse_msg": state["nurse_msg"]
    }
    response = (doctor_prompt | llm).invoke(inputs)
    if isinstance(response, str):
        return {**state, "doctor_msg": response, "iteration": state.get("iteration", 0) + 1}
    return {
        **state,
        "doctor_msg": response.content,
        "iteration": state.get("iteration", 0) + 1
    }

# Define LangGraph workflow
workflow = StateGraph(state_schema=dict)
workflow.add_node("Nurse", nurse_step)
workflow.add_node("Doctor", doctor_step)
workflow.set_entry_point("Nurse")
workflow.add_edge("Nurse", "Doctor")
workflow.add_conditional_edges(
    "Doctor",
    lambda state: "Nurse" if state.get("iteration", 0) < 2 else END
)

app = workflow.compile()

def run_triage_workflow(note: str) -> dict:
    result = app.invoke({"note": note})
    return {
        "esi": result.get("nurse_msg", "N/A"),
        "diagnosis": result.get("doctor_msg", "N/A"),
        "iterations": result.get("iteration", 0)
    }

if __name__ == "__main__":
    test_note = "45-year-old with chest pain radiating to arm, shortness of breath."
    output = run_triage_workflow(test_note)
    print("Triage Result:", output)
