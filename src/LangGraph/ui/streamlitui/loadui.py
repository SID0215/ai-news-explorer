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
ENABLE_GDELT = os.getenv("ENABLE_GDELT")


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

                self.user_controls["GROQ_API_KEY"] = GROQ_API_KEY

            # ---- Usecase selection ----
            self.user_controls["USE_CASE_OPTIONS"] = st.selectbox(
                "Select Usecase", usecase_options
            )

            # ----------------- NEWS CONTROLS -----------------
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

            # If radio was hidden (chatbot usecase), ensure we still have a NEWS_TYPE
            if "NEWS_TYPE" not in self.user_controls:
                self.user_controls["NEWS_TYPE"] = st.session_state.get(
                    "NEWS_TYPE", "news"
                )

            # Set Tavily key for anything that needs web search
            if self.user_controls["USE_CASE_OPTIONS"] in (
                "News",
                "Chatbot with tavily search",
            ):
                if TAVILY_API_KEY:
                    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
                    self.user_controls["TAVILY_API_KEY"] = TAVILY_API_KEY
                else:
                    st.warning(
                        "TAVILY_API_KEY is not set in .env; web search may not work."
                    )

            # You can also expose other API keys if needed
            self.user_controls["NEWS_DATA_API_KEY"] = NEWS_DATA_API_KEY
            self.user_controls["GUARDIAN_API_KEY"] = GUARDIAN_API_KEY
            self.user_controls["ENABLE_GDELT"] = ENABLE_GDELT

            # ---- News Explorer (time frame + date) ----
            if self.user_controls["USE_CASE_OPTIONS"] == "News":
                st.subheader("News Explorer")

                time_frame_label = st.selectbox(
                    "Select Time Frame", ["Today", "Weekly", "Monthly"], index=0
                )
                tf_value = time_frame_label.lower()  # "today", "weekly", "monthly"

                st.session_state["timeframe"] = tf_value
                st.session_state["timeframe_label"] = time_frame_label
                self.user_controls["TIMEFRAME"] = tf_value

                today = date.today()
                default_date = st.session_state.get("selected_date", today)
                selected_date = st.date_input(
                    "Select a date (optional)",
                    value=default_date,
                    max_value=today,
                )
                st.session_state["selected_date"] = selected_date

                if st.button("Fetch Latest News", use_container_width=True):
                    st.session_state["IsFetchButtonClicked"] = True

        return self.user_controls
