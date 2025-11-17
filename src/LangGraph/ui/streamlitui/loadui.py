import os
import uuid
from datetime import date

import streamlit as st
from dotenv import load_dotenv

from src.LangGraph.ui.ui_config import Config

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_DATA_API_KEY = os.getenv("NEWS_DATA_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
ENABLE_GDELT=os.getenv("ENABLE_GDELT")

class LoadStreamLitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit(self):
        # ----------------- PAGE SETUP -----------------
        st.set_page_config(
            page_title=self.config.get_title(),
            layout="wide",
        )
        st.header(self.config.get_title())

        # Initialise common session state values
        st.session_state.setdefault("IsFetchButtonClicked", False)
        st.session_state.setdefault("IsFetchAIButtonClicked", False)
        st.session_state.setdefault("timeframe", "today")

        if "thread_id" not in st.session_state:
            st.session_state["thread_id"] = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # ----------------- SIDEBAR -----------------
        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # ---- LLM selection ----
            self.user_controls["select_llm"] = st.selectbox(
                "Select LLM", llm_options
            )

            if self.user_controls["select_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox(
                    "Select Model", model_options
                )

                if self.user_controls["selected_groq_model"] == "other":
                    self.user_controls["other_model"] = st.text_input(
                        "Enter your Groq model name", type="default"
                    )
                    if not self.user_controls["other_model"]:
                        st.warning("Please enter your custom model name to proceed.")

                # expose key for graph builder
                self.user_controls["GROQ_API_KEY"] = GROQ_API_KEY

            # ---- Usecase selection ----
            self.user_controls["USE_CASE_OPTIONS"] = st.selectbox(
                "Select Usecase", usecase_options
            )

            # ----------------- NEWS CONTROLS -----------------
            # we keep NEWS_TYPE in session so other modules can read it
            if self.user_controls["USE_CASE_OPTIONS"] != "Chatbot with tavily search":
                st.markdown("### 📰 Choose News Type")

                self.user_controls["NEWS_TYPE"] = st.radio(
                    "Choose News Category",
                    ["news", "general", "finance", "movies", "sports", "business", "tech"],
                    horizontal=True,
                )

                st.session_state["NEWS_TYPE"] = self.user_controls["NEWS_TYPE"]
                st.info(
                    f"Selected News Type: **{self.user_controls['NEWS_TYPE'].capitalize()}**"
                )

            # Only set Tavily key when in News mode
            if self.user_controls["USE_CASE_OPTIONS"] == "News":
                os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
                self.user_controls["TAVILY_API_KEY"] = TAVILY_API_KEY

            # ---- News Explorer (time frame + date) ----
            if self.user_controls["USE_CASE_OPTIONS"] == "News":
                st.subheader("News Explorer")

                # Time frame selector (Today / Weekly / Monthly)
                time_frame_label = st.selectbox(
                    "Select Time Frame", ["Today", "Weekly", "Monthly"], index=0
                )
                tf_value = time_frame_label.lower()  # "today", "weekly", "monthly"

                st.session_state["timeframe"] = tf_value
                st.session_state["timeframe_label"] = time_frame_label
                self.user_controls["TIMEFRAME"] = tf_value

                # Optional date picker; user cannot select beyond today
                today = date.today()
                default_date = st.session_state.get("selected_date", today)
                selected_date = st.date_input(
                    "Select a date (optional)",
                    value=default_date,
                    max_value=today,
                )
                st.session_state["selected_date"] = selected_date

                # Fetch button
                if st.button("Fetch Latest News", use_container_width=True):
                    st.session_state["IsFetchButtonClicked"] = True

        return self.user_controls
