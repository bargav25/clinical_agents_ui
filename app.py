import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserConfig:
    id: int
    name: str
    email: str
    age: int
    gender: str
    user_type: str
    created_at: str
    updated_at: str

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: Optional[datetime] = None

class Config:
    API_URL = os.getenv("API_URL", "https://clinical-agents-333016757590.us-central1.run.app/api/v1")
    PAGE_TITLE = "AI Triage System"
    PAGE_ICON = "🏥"
    
    # UI Constants
    SIDEBAR_WIDTH = 300
    CHAT_HEIGHT = 500
    
    # Colors
    PRIMARY_COLOR = "#1f77b4"
    SUCCESS_COLOR = "#2ca02c"
    WARNING_COLOR = "#ff7f0e"
    ERROR_COLOR = "#d62728"

# ─────────────────────────────────────────────────────────────────────────────
# Session State Manager
# ─────────────────────────────────────────────────────────────────────────────

class SessionStateManager:
    @staticmethod
    def init_state():
        """Initialize session state with default values"""
        defaults = {
            "user_type": "patient",
            "auth_done": False,
            "user_id": None,
            "user_data": None,
            "messages": [],
            "notes": [],
            "chat_active": False,
            "finished": False,
            "current_assessment_id": None,
            "show_help": False
        }
        
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val
    
    @staticmethod
    def get_user_config() -> Optional[UserConfig]:
        """Get current user configuration"""
        if st.session_state.auth_done and st.session_state.user_data:
            data = st.session_state.user_data
            return UserConfig(
                id=data["id"],
                name=data["name"],
                email=data["email"],
                age=data["age"],
                gender=data["gender"],
                user_type=data["user_type"],
                created_at=data["created_at"],
                updated_at=data["updated_at"]
            )
        return None
    
    @staticmethod
    def reset_chat():
        """Reset chat state"""
        st.session_state.messages = []
        st.session_state.chat_active = False
        st.session_state.finished = False
        st.session_state.current_assessment_id = None

# ─────────────────────────────────────────────────────────────────────────────
# API Service Layer
# ─────────────────────────────────────────────────────────────────────────────

class APIService:
    @staticmethod
    def login_user(name: str, email: str, age: int, gender: str, user_type: str) -> Tuple[bool, Dict]:
        """Login user with the new API structure"""
        try:
            payload = {
                "name": name,
                "email": email,
                "age": age,
                "gender": gender,
                "user_type": user_type
            }
            
            resp = requests.post(
                f"{Config.API_URL}/users/login",
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            return True, resp.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    @staticmethod
    def get_user_by_id(user_id: int) -> Tuple[bool, Dict]:
        """Get user information by ID"""
        try:
            resp = requests.get(f"{Config.API_URL}/users/{user_id}", timeout=10)
            resp.raise_for_status()
            return True, resp.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    @staticmethod
    def fetch_assessments() -> List[Dict]:
        """Fetch all assessments from API"""
        try:
            resp = requests.get(f"{Config.API_URL}/assessments", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"Failed to fetch assessments: {str(e)}")
            return []
    
    @staticmethod
    def send_chat_message(message: str, history: List[Dict], patient_id: int) -> Tuple[bool, Dict]:
        """Send chat message to triage API"""
        try:
            payload = {
                "message": message,
                "history": history,
                "patient_id": patient_id
            }
            
            resp = requests.post(
                f"{Config.API_URL}/triage/chat",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            return True, resp.json()
        except Exception as e:
            return False, {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────────────────────────────────────

class UIComponents:
    @staticmethod
    def render_header():
        """Render main header with branding"""
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #1f77b4, #2ca02c); 
                    border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">🏥 AI Triage System</h1>
            <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Intelligent healthcare assessment at your fingertips
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_auth() -> bool:
        """Render sidebar authentication section"""
        with st.sidebar:
            st.markdown("### 👤 Authentication")
            
            # User type selection
            user_type = st.selectbox(
                "Select your role:",
                ["patient", "staff"],
                key="user_type_select",
                help="Choose whether you're a patient seeking assessment or medical staff"
            )
            st.session_state.user_type = user_type
            
            # Sign in form
            with st.form("signin_form"):
                st.markdown("**Sign In / Register**")
                name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email address")
                
                # Additional fields for patients
                if user_type == "patient":
                    col1, col2 = st.columns(2)
                    with col1:
                        age = st.number_input("Age", min_value=1, max_value=120, value=25)
                    with col2:
                        gender = st.selectbox("Gender", ["male", "female", "other"])
                else:
                    age = 30  # Default for staff
                    gender = "not_specified"
                
                submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)
                
                if submitted:
                    if name.strip() and email.strip():
                        with st.spinner("🔄 Signing in..."):
                            success, response = APIService.login_user(
                                name.strip(), 
                                email.strip(), 
                                age, 
                                gender, 
                                user_type
                            )
                            
                            if success:
                                st.session_state.user_data = response
                                st.session_state.user_id = response["id"]
                                st.session_state.auth_done = True
                                st.success("✅ Successfully signed in!")
                                st.rerun()
                            else:
                                st.error(f"❌ Login failed: {response.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Please enter both name and email")
            
            # Show current user info
            if st.session_state.auth_done and st.session_state.user_data:
                st.markdown("---")
                st.markdown("**Current User:**")
                user_data = st.session_state.user_data
                st.info(f"""
                👤 **{user_data['name']}**
                📧 {user_data['email']}
                🏷️ {user_data['user_type'].title()}
                🆔 ID: {user_data['id']}
                """)
                
                if st.button("🚪 Sign Out", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        
        return st.session_state.auth_done
    
    @staticmethod
    def render_chat_interface(user_config: UserConfig):
        """Render patient chat interface"""
        st.markdown("### 💬 Chat Assessment")
        
        # Chat container
        chat_container = st.container(height=Config.CHAT_HEIGHT)
        
        with chat_container:
            # Display chat history
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if not st.session_state.chat_active:
                if st.button("🚀 Start New Assessment", use_container_width=True, type="primary"):
                    UIComponents._start_new_assessment(user_config)
        
        with col2:
            if st.session_state.chat_active and st.button("🔄 Reset Chat", use_container_width=True):
                SessionStateManager.reset_chat()
                st.rerun()
        
        with col3:
            if st.button("❓ Help", use_container_width=True):
                st.session_state.show_help = not st.session_state.get("show_help", False)
        
        # Help section
        if st.session_state.get("show_help", False):
            st.markdown("""
            ---
            **ℹ️ How to use the AI Triage System:**
            
            1. **Start Assessment**: Click 'Start New Assessment' to begin
            2. **Describe Symptoms**: Be detailed about your symptoms, when they started, and their severity
            3. **Answer Questions**: The AI will ask follow-up questions to better understand your condition
            4. **Get Results**: Receive your triage level and recommended next steps
            
            **Tips for better results:**
            - Be honest and specific about your symptoms
            - Include timeline information (when symptoms started)
            - Mention any relevant medical history
            - Don't hesitate to ask for clarification
            """)
        
        # Chat input
        if st.session_state.chat_active and not st.session_state.finished:
            if user_input := st.chat_input("💭 Describe your symptoms or ask a question..."):
                UIComponents._handle_user_message(user_input, user_config)
    
    @staticmethod
    def _start_new_assessment(user_config: UserConfig):
        """Start a new triage assessment"""
        st.session_state.chat_active = True
        st.session_state.messages = []
        st.session_state.finished = False
        
        with st.spinner("🔄 Starting your assessment..."):
            success, response = APIService.send_chat_message("", [], user_config.id)
            
            if success:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"]
                })
                st.rerun()
            else:
                st.error(f"❌ Failed to start assessment: {response.get('error', 'Unknown error')}")
                st.session_state.chat_active = False
    
    @staticmethod
    def _handle_user_message(user_input: str, user_config: UserConfig):
        """Handle user message in chat"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤔 AI is analyzing your response..."):
            # Convert messages to API format
            history = [{"role": msg["role"], "content": msg["content"]} 
                      for msg in st.session_state.messages]
            
            success, response = APIService.send_chat_message(user_input, history, user_config.id)
            
            if success:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"]
                })
                
                if response.get("finished", False):
                    st.session_state.chat_active = False
                    st.session_state.finished = True
                    st.success("✅ Assessment completed successfully!")
                    st.balloons()
                
                st.rerun()
            else:
                st.error(f"❌ Error: {response.get('error', 'Unknown error')}")

class StaffDashboard:
    @staticmethod
    def render_dashboard(user_config: UserConfig):
        """Render staff dashboard"""
        st.markdown("### 📊 Staff Dashboard")
        
        # Fetch data
        with st.spinner("📥 Loading assessment data..."):
            assessments = APIService.fetch_assessments()
        
        if not assessments:
            st.info("📭 No assessments available yet.")
            return
        
        df = pd.DataFrame(assessments)
        df["created_at"] = pd.to_datetime(df["created_at"])
        
        # Dashboard metrics
        StaffDashboard._render_metrics(df)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            StaffDashboard._render_esi_distribution(df)
        with col2:
            StaffDashboard._render_timeline_chart(df)
        
        # Data table
        StaffDashboard._render_assessments_table(df)
    
    @staticmethod
    def _render_metrics(df: pd.DataFrame):
        """Render key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📋 Total Assessments",
                len(df),
                delta=None
            )
        
        with col2:
            avg_esi = df['esi_level'].mean()
            st.metric(
                "⚡ Avg ESI Level",
                f"{avg_esi:.1f}",
                delta=None
            )
        
        with col3:
            emergency_cases = len(df[df['esi_level'] <= 2])
            st.metric(
                "🚨 Emergency Cases",
                emergency_cases,
                delta=f"{(emergency_cases/len(df)*100):.1f}% of total"
            )
        
        with col4:
            latest = df['created_at'].max()
            hours_ago = (datetime.now() - latest.replace(tzinfo=None)).total_seconds() / 3600
            st.metric(
                "🕐 Last Assessment",
                f"{hours_ago:.1f}h ago",
                delta=None
            )
    
    @staticmethod
    def _render_esi_distribution(df: pd.DataFrame):
        """Render ESI level distribution chart"""
        st.markdown("**🎯 ESI Level Distribution**")
        
        esi_counts = df['esi_level'].value_counts().sort_index()
        
        colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a']
        
        fig = px.bar(
            x=esi_counts.index,
            y=esi_counts.values,
            color=esi_counts.index,
            color_continuous_scale='RdYlGn_r',
            title="Distribution by ESI Level"
        )
        
        fig.update_layout(
            xaxis_title="ESI Level",
            yaxis_title="Count",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_timeline_chart(df: pd.DataFrame):
        """Render assessments timeline"""
        st.markdown("**📈 Assessment Timeline**")
        
        # Group by date
        df_daily = df.groupby(df['created_at'].dt.date).size().reset_index()
        df_daily.columns = ['date', 'count']
        
        fig = px.line(
            df_daily,
            x='date',
            y='count',
            title="Daily Assessment Volume",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Assessments",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_assessments_table(df: pd.DataFrame):
        """Render assessments data table"""
        st.markdown("**📋 Recent Assessments**")
        
        # Prepare display dataframe
        display_df = df.copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add user information by fetching user details
        if not display_df.empty:
            # Create a dictionary to store user info to avoid multiple API calls
            user_info_cache = {}
            user_names = []
            user_emails = []
            
            for user_id in display_df['user_id']:
                if user_id not in user_info_cache:
                    success, user_data = APIService.get_user_by_id(user_id)
                    if success:
                        user_info_cache[user_id] = user_data
                    else:
                        user_info_cache[user_id] = {"name": "Unknown", "email": "Unknown"}
                
                user_info = user_info_cache[user_id]
                user_names.append(user_info.get("name", "Unknown"))
                user_emails.append(user_info.get("email", "Unknown"))
            
            display_df['patient_name'] = user_names
            display_df['patient_email'] = user_emails
        
        # Select columns to display
        columns = ["id", "patient_name", "patient_email", "esi_level", "diagnosis", "notes", "created_at"]
        
        # Sort by creation time (newest first)
        display_df = display_df.sort_values('created_at', ascending=False)
        
        st.dataframe(
            display_df[columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "Assessment ID",
                "patient_name": "Patient Name",
                "patient_email": "Patient Email",
                "esi_level": st.column_config.NumberColumn(
                    "ESI Level",
                    help="Emergency Severity Index (1=Most urgent, 5=Least urgent)",
                    min_value=1,
                    max_value=5,
                    format="%d"
                ),
                "diagnosis": "Diagnosis",
                "notes": "Notes",
                "created_at": "Created At"
            }
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        layout="wide",
        page_icon=Config.PAGE_ICON,
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    SessionStateManager.init_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton > button {
        border-radius: 20px;
        border: none;
        font-weight: 600;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render header
    UIComponents.render_header()
    
    # Handle authentication
    if not UIComponents.render_sidebar_auth():
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>👋 Welcome to AI Triage System</h3>
            <p>Please sign in using the sidebar to get started with your health assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Get user configuration
    user_config = SessionStateManager.get_user_config()
    
    # Route to appropriate interface
    if user_config.user_type == "patient":
        UIComponents.render_chat_interface(user_config)
    else:
        StaffDashboard.render_dashboard(user_config)

if __name__ == "__main__":
    main()