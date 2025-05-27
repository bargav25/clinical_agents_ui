# triage_ai_assistant/app.py
import streamlit as st
import sqlite3
import re
import logging
from datetime import datetime
from agents.triageagent import run_triage_workflow
from agents.nursebot import handle_chat, NURSEBOT_SYSINT, WELCOME_MSG, llm_with_tools
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
import pandas as pd
# Load environment variables
load_dotenv()

print("Loading environment variables...")
# print("GOOGLE API KEY:", os.getenv("GOOGLE_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "triage.db"

# ─────────────────────────────────────────────────────────────────────────────
# Database Functions
# ─────────────────────────────────────────────────────────────────────────────
def create_table():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        create_script = '''
        CREATE TABLE IF NOT EXISTS patient_assessments(
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_email TEXT NOT NULL,
            patient_notes TEXT NOT NULL,
            esi_level INTEGER NOT NULL,
            diagnosis TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )'''
        cur.execute(create_script)
        conn.commit()
        logger.info("Table patient_assessments created successfully or already exists")
    except Exception as error:
        logger.error(f"Error creating table: {error}")
    finally:
        cur.close()
        conn.close()

def insert_assessment(notes: str, esi_level: int, diagnosis: str, patient_name: str, patient_email: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        insert_script = '''
        INSERT INTO patient_assessments (patient_notes, esi_level, diagnosis, patient_name, patient_email) 
        VALUES (?, ?, ?, ?, ?)
        '''
        insert_value = (notes, esi_level, diagnosis, patient_name, patient_email)
        cur.execute(insert_script, insert_value)
        conn.commit()
        inserted_id = cur.lastrowid
        return inserted_id
    except Exception as error:
        logger.error(f"Database error during insert: {error}")
        raise
    finally:
        cur.close()
        conn.close()

def view_assessments():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM patient_assessments ORDER BY created_at DESC")
        return cur.fetchall()
    except Exception as error:
        logger.error(f"Error fetching assessments: {error}")
        return []
    finally:
        cur.close()
        conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# Core Application Logic
# ─────────────────────────────────────────────────────────────────────────────
def is_prompt_injection(input_text: str) -> bool:
    suspicious_phrases = [
        r"ignore\s+(all|previous|above)\s+instructions",
        r"disregard\s+(this|that|everything)",
        r"forget\s+.*\bprevious\b",
        r"act\s+as\s+.*",
        r"(system|you)\s+are\s+now",
        r"you\s+are\s+no\s+longer\s+an\s+AI"
    ]
    input_text_lower = input_text.lower()
    for pattern in suspicious_phrases:
        if re.search(pattern, input_text_lower):
            return True
    return False

def extract_esi_level(esi_str: str) -> int:
    try:
        match = re.search(r'ESI\s*(\d+)|Level\s*(\d+)', esi_str, re.IGNORECASE)
        if match:
            number = next(num for num in match.groups() if num is not None)
            return int(number)
        return 3
    except Exception:
        return 3

def handle_chat_interaction(message: str, history: list, patient_name: str, patient_email: str):
    if not history:
        return {"response": WELCOME_MSG, "finished": False, "notes": []}
    
    messages = [SystemMessage(content=NURSEBOT_SYSINT[1])]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    response = llm_with_tools.invoke(messages)
    notes = []
    finished = False
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            if call["name"] == "take_note" and "text" in call["args"]:
                notes.append(call["args"]["text"])
                finished = True
    
    if finished:
        combined_note = "\n".join(notes)
        print(f"Combined Note: {combined_note}")
        triage_result = run_triage_workflow(combined_note)
        esi_level = extract_esi_level(triage_result['esi'])
        
        try:
            insert_assessment(
                notes=combined_note,
                esi_level=esi_level,
                diagnosis=triage_result['diagnosis'],
                patient_name=patient_name,
                patient_email=patient_email
            )
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
        
        return {
            "response": f"Triage Assessment Complete!\nESI Level: {esi_level}\nDiagnosis: {triage_result['diagnosis']}",
            "finished": True,
            "notes": notes
        }
    
    return {
        "response": response.content,
        "finished": False,
        "notes": notes
    }

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI Components
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "role": None,
        "auth_done": False,
        "user_name": "",
        "user_email": "",
        "messages": [],
        "notes": [],
        "chat_active": False,
        "finished": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def patient_interface():
    st.header(f"🩺 Patient Assessment ({st.session_state.user_name})")
    
    if not st.session_state.chat_active:
        if st.button("Start New Triage Assessment"):
            st.session_state.chat_active = True
            st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if st.session_state.chat_active:
        if user_input := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.write(user_input)
                    
            with st.spinner("Processing..."):
                result = handle_chat_interaction(
                    message=user_input,
                    history=st.session_state.messages,
                    patient_name=st.session_state.user_name,
                    patient_email=st.session_state.user_email
                )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })

                with st.chat_message("assistant"):
                    st.write(result["response"])
                            
                if result["finished"]:
                    st.session_state.chat_active = False
                    st.success("✅ Assessment Complete")

def staff_interface():
    st.header(f"👩‍⚕️ Staff Dashboard ({st.session_state.user_name})")
    
    assessments = view_assessments()
    if not assessments:
        st.info("No assessments yet.")
        return
    
    df = pd.DataFrame(assessments, columns=[
        "assessment_id", "patient_name", "patient_email",
        "patient_notes", "esi_level", "diagnosis", "created_at"
    ])

    df['created_at'] = pd.to_datetime(df['created_at'])
    
    st.markdown(f"**Logged in as:** {st.session_state.user_name} ({st.session_state.user_email})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Assessments", len(df))
    c2.metric("Avg. ESI Level", round(df['esi_level'].mean(), 1))
    c3.metric("Last Assessment", df.iloc[0]['created_at'].strftime("%Y-%m-%d %H:%M"))
    
    st.dataframe(df[[
        "assessment_id", "patient_name", "patient_email",
        "esi_level", "created_at"
    ]], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main App Configuration
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AI Triage System", layout="wide", page_icon="🏥")
    init_session_state()
    create_table()
    
    with st.sidebar:
        st.title("🏥 AI Triage System")
        st.session_state.role = st.radio("I am a:", ["Patient", "Staff"])
        
        if not st.session_state.auth_done:
            st.subheader("Sign In")
            name = st.text_input("Name")
            email = st.text_input("Email")
            
            if st.button("Sign In"):
                if name and email:
                    st.session_state.user_name = name
                    st.session_state.user_email = email
                    st.session_state.auth_done = True
                    st.rerun()
                else:
                    st.error("Name and email required.")
        else:
            st.markdown(f"**Signed in as:** {st.session_state.user_name} ({st.session_state.user_email})")
    
    if not st.session_state.auth_done:
        st.write("Please sign in using the sidebar.")
        return
    
    if st.session_state.role == "Patient":
        patient_interface()
    else:
        staff_interface()

if __name__ == "__main__":
    main()