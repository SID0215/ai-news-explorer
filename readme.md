# LangGraph AI News Bot

A **LangGraph-powered AI dashboard** for chatting with an AI, fetching real-time web search results, and generating AI news summaries. Built with modular design to easily extend for new AI agent use cases.

---

## 🚀 Features

- 🤖 **Basic Chatbot** – Interact with an AI assistant for general conversations.
- 🔍 **Chatbot with Tavily Search** – Real-time web search integration powered by Tavily API.
- 📰 **AI News Summarizer** – Generate daily or weekly or monthly AI news summaries.
- 🌐 **General News Explorer** – Get news across various topics.
- ⚡ **Modular Architecture** – Organized into nodes, tools, state, and UI components.
- 🔑 **API Key Management** – Secure `.env` configuration for sensitive keys.

---


## 📂 Project Structure

```plaintext
.
├── AINews/                     # Stores AI news summaries
│   ├── daily_summary.md
│   ├── weekly_summary.md
│
├── src/                        # Source folder
│   ├── LangGraph/              # Main application modules
│   │   ├── graph/              # Graph definitions and workflows
│   │   ├── llms/               # LLM integration logic (Groq, DeepSeek, etc.)
│   │   ├── nodes/              # LangGraph nodes (AI news, chatbot, Tavily search)
│   │   ├── state/              # State management logic
│   │   ├── tools/              # Utility tools (news fetchers, summarizers)
│   │   └── ui/                 # Streamlit UI components
│   └── __init__.py
│
├── screenshots/                # App screenshots
│   ├── screenshot-dashboard.png
│   ├── screenshot-tavily-chat.png
│   ├── screenshot-ai-news.png
│   ├── screenshot-news.png
│
├── main.py                     # Main Streamlit entry point
├── requirements.txt            # Python dependencies
├── .env                        # API keys (ignored by Git)
├── .gitignore                  # Ignored files
└── README.md                   # Documentation

---


## ⚙️ Project Setup
    1️⃣ Clone the repository
        -- git clone https://github.com/gk-j/langgraph-ai-news-bot.git
        -- cd langgraph-ai-news-bot
    2️⃣ Create and activate a Conda environment
        -- conda create -n langgraph-news python=3.12
        -- conda activate langgraph-news
    3️⃣ Install dependencies
        -- pip install -r requirements.txt
    4️⃣ Run the app
        -- streamlit run app.py