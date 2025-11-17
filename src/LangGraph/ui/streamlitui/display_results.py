import os
import re
import json
from datetime import date, datetime, timedelta

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.LangGraph.state.state import State

# Optional: used to fetch article images / video from the news URL
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None


# -------------------------------------------------------------------
# HELPERS: PARSE MARKDOWN → STRUCTURED SECTIONS
# -------------------------------------------------------------------
def parse_news_markdown_grouped(markdown_text: str):
    """
    Parse a markdown summary file of the form:

        # Today News Summary

        ### 2025-11-16
        - **Headline A**: 2–3 sentence summary... [Read full story](https://link1)

        ### 2025-11-15
        - **Headline B**: Another summary... [Read full story](https://link2)

    Returns:
        [
          {"date": "2025-11-16", "articles": [ {title, summary, url}, ... ]},
          {"date": "2025-11-15", "articles": [ ... ]},
          ...
        ]
    """
    sections = []
    current_date = None
    current_articles = []
    seen_urls = set()

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()

        # Ignore main H1 heading etc.
        if not line or line.startswith("# "):
            continue

        # New date heading
        if line.startswith("### "):
            if current_date and current_articles:
                sections.append({"date": current_date, "articles": current_articles})
            current_date = line.replace("###", "").strip()
            current_articles = []
            continue

        # Bullet line: - **Title**: Summary text [Read full story](URL)
        if line.startswith("- "):
            m = re.search(
                r"-\s+\*\*(.+?)\*\*:\s*(.+?)\s*\[.*?\]\((https?://[^\)]+)\)",
                line,
            )
            if not m:
                continue

            title = m.group(1).strip()
            summary = m.group(2).strip()
            url = m.group(3).strip()

            norm_url = url.split("?", 1)[0].strip()

            # Remove duplicates across the whole file by URL
            if norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)

            current_articles.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": url,
                }
            )

    # Last section
    if current_date and current_articles:
        sections.append({"date": current_date, "articles": current_articles})

    # Fallback: if there were no ### headings, treat as "No Date"
    if not sections:
        flat_articles = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue

            m = re.search(
                r"-\s+\*\*(.+?)\*\*:\s*(.+?)\s*\[.*?\]\((https?://[^\)]+)\)",
                line,
            )
            if not m:
                continue

            title = m.group(1).strip()
            summary = m.group(2).strip()
            url = m.group(3).strip()

            norm_url = url.split("?", 1)[0].strip()
            if norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)

            flat_articles.append(
                {"title": title, "summary": summary, "url": url}
            )

        if flat_articles:
            sections.append({"date": "No Date", "articles": flat_articles})

    return sections


# -------------------------------------------------------------------
# HELPERS: MEDIA (IMAGE + VIDEO)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_article_media(url: str):
    """
    Try to fetch an image URL and a video URL from a news article page
    using OpenGraph/Twitter meta tags.

    Returns:
        {"image": <url or None>, "video": <url or None>}
    """
    if not url or not requests or not BeautifulSoup:
        return {"image": None, "video": None}

    try:
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (genai-news-app)"},
        )
        resp.raise_for_status()
    except Exception:
        return {"image": None, "video": None}

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return {"image": None, "video": None}

    image_url = None
    video_url = None

    # Common OpenGraph / Twitter image tags
    for attr, key in [
        ("property", "og:image"),
        ("name", "og:image"),
        ("property", "twitter:image"),
        ("name", "twitter:image"),
    ]:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            image_url = tag["content"]
            break

    # Look for video-related meta tags
    for attr, key in [
        ("property", "og:video"),
        ("name", "og:video"),
        ("property", "twitter:player"),
        ("name", "twitter:player"),
    ]:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            video_url = tag["content"]
            break

    return {"image": image_url, "video": video_url}


def _get_fallback_image(news_type: str) -> str:
    """Static fallback images if we can't fetch from the article URL."""
    CATEGORY_FALLBACK_IMAGES = {
        "sports": "https://images.pexels.com/photos/399187/pexels-photo-399187.jpeg",
        "finance": "https://images.pexels.com/photos/210607/pexels-photo-210607.jpeg",
        "business": "https://images.pexels.com/photos/37347/office-freelancer-computer-business-37347.jpeg",
        "movies": "https://images.pexels.com/photos/799137/pexels-photo-799137.jpeg",
        "tech": "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg",
        "news": "https://images.pexels.com/photos/2318555/pexels-photo-2318555.jpeg",
        "general": "https://images.pexels.com/photos/2619490/pexels-photo-2619490.jpeg",
    }
    return CATEGORY_FALLBACK_IMAGES.get(
        news_type.lower(), CATEGORY_FALLBACK_IMAGES["general"]
    )


# -------------------------------------------------------------------
# HELPERS: DATE / RANGE FILTERING
# -------------------------------------------------------------------
def _get_selected_date() -> date | None:
    """Pull selected date from session_state (from LoadStreamLitUI)."""
    sd = st.session_state.get("selected_date")
    if isinstance(sd, date):
        return sd
    if isinstance(sd, str):
        try:
            return datetime.fromisoformat(sd).date()
        except Exception:
            return None
    return None


def filter_sections_by_selected_date(sections, timeframe: str):
    """
    Keep only sections whose date falls within the selected date range:

    - Today: that exact date
    - Weekly: Monday–Sunday week containing selected date
    - Monthly: whole calendar month of selected date
    """
    selected = _get_selected_date()
    if not selected:
        return sections

    tf = timeframe.lower()
    if tf.startswith("today"):
        start = end = selected
    elif tf.startswith("week"):
        # Monday–Sunday week
        start = selected - timedelta(days=selected.weekday())
        end = start + timedelta(days=6)
    elif tf.startswith("month"):
        start = selected.replace(day=1)
        if selected.month == 12:
            end = selected.replace(year=selected.year + 1, month=1, day=1) - timedelta(
                days=1
            )
        else:
            end = selected.replace(month=selected.month + 1, day=1) - timedelta(days=1)
    else:
        return sections

    filtered = []
    today = date.today()

    for section in sections:
        try:
            sec_date = datetime.fromisoformat(section.get("date", "")).date()
        except Exception:
            # If we can't parse, keep it (could be "Latest" or "No Date")
            filtered.append(section)
            continue

        # Clamp any future dates
        if sec_date > today:
            continue

        if start <= sec_date <= end:
            filtered.append(section)

    return filtered


# -------------------------------------------------------------------
# RENDERING: ARTICLE GRID
# -------------------------------------------------------------------
def render_article_grid(articles, news_type: str):
    """
    Render a responsive grid of tiles for all articles of a single date.
    Each tile:
        - category tag (Movies / Sports / Tech / etc.)
        - image or video thumbnail
        - title
        - 60–150 word summary
        - "Read full story →" link
    """
    fallback_img = _get_fallback_image(news_type)
    tag_label = news_type.capitalize()

    st.markdown(
        """
        <style>
        .news-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1.0rem;
            margin-bottom: 1.8rem;
        }
        .news-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10);
            transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .news-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
        }
        .news-tag {
            display: inline-block;
            padding: 0.10rem 0.55rem;
            border-radius: 999px;
            font-size: 0.70rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            background: #eff6ff;
            color: #1d4ed8;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        .news-media {
            width: 100%;
            height: 170px;
            border-radius: 16px;
            object-fit: cover;
            margin-bottom: 0.9rem;
        }
        .news-title {
            font-size: 1.0rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.4rem;
        }
        .news-summary {
            font-size: 0.9rem;
            color: #4b5563;
            line-height: 1.4;
            margin-bottom: 0.7rem;
        }
        .news-link {
            margin-top: auto;
            font-size: 0.9rem;
            font-weight: 600;
            color: #2563eb;
            text-decoration: none;
        }
        .news-link:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cards_html = []
    for art in articles:
        title = art.get("title", "Untitled")
        summary = art.get("summary", "Tap to read the full story →")
        url = art.get("url", "#")

        # Try to get media from article itself
        media = fetch_article_media(url)
        img_url = media.get("image") or fallback_img
        video_url = media.get("video")

        if video_url:
            # If looks like direct video file → <video>, otherwise <iframe>
            if any(video_url.lower().endswith(ext) for ext in [".mp4", ".webm", ".ogg"]):
                media_html = f"""
                <video class="news-media" controls preload="metadata">
                    <source src="{video_url}">
                    Your browser does not support the video tag.
                </video>
                """
            else:
                media_html = f"""
                <iframe class="news-media" src="{video_url}"
                        frameborder="0" allowfullscreen></iframe>
                """
        else:
            media_html = f'<img src="{img_url}" class="news-media" />'

        card = f"""
        <div class="news-card">
            <div class="news-tag">{tag_label}</div>
            {media_html}
            <div class="news-title">{title}</div>
            <div class="news-summary">{summary}</div>
            <a href="{url}" target="_blank" class="news-link">
                Read full story →
            </a>
        </div>
        """
        cards_html.append(card)

    st.markdown('<div class="news-grid">', unsafe_allow_html=True)
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# RENDERING: FULL NEWS PAGE
# -------------------------------------------------------------------
def render_news_sections(sections, news_type: str, timeframe: str):
    """
    Render the full news view.
    """
    tf = timeframe.capitalize()
    label = st.session_state.get("timeframe_label", tf)

    if timeframe.lower().startswith("today"):
        heading = "Today News Summary"
    elif timeframe.lower().startswith("week"):
        heading = "Weekly News Summary"
    else:
        heading = "Monthly News Summary"

    st.markdown(f"## {heading}")

    selected = _get_selected_date()
    if selected:
        if timeframe.lower().startswith("week"):
            week_start = selected - timedelta(days=selected.weekday())
            week_end = week_start + timedelta(days=6)
            st.caption(f"Week of {week_start.isoformat()} to {week_end.isoformat()}")
        elif timeframe.lower().startswith("month"):
            st.caption(selected.strftime("%B %Y"))
        else:
            st.caption(selected.isoformat())
    else:
        # Fallback: today
        st.caption(date.today().isoformat())

    st.write(f"**Selected News Type:** {news_type.capitalize()}")

    if not sections:
        st.info("No news summaries found for this selection yet.")
        return

    # Filter sections by selected date + timeframe (day/week/month)
    sections = filter_sections_by_selected_date(sections, timeframe)
    if not sections:
        st.warning("No news available for the selected date range.")
        return

    # Render each date section
    for section in sections:
        day = section.get("date", "Latest")
        articles = section.get("articles", [])
        if not articles:
            continue

        st.markdown(f"### {day}")
        render_article_grid(articles, news_type)


# -------------------------------------------------------------------
# MAIN CLASS
# -------------------------------------------------------------------
class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message, thread_id):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.thread_id = thread_id

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # --------------------------------------------------------------
        # 1) BASIC CHATBOT
        # --------------------------------------------------------------
        if usecase == "Basic Chatbot":
            for msg in st.session_state["messages"]:
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                with st.chat_message(role):
                    st.write(msg.content)

            if user_message:
                user_msg = HumanMessage(content=user_message)
                st.session_state["messages"].append(user_msg)
                with st.chat_message("user"):
                    st.write(user_message)

                state = State(messages=st.session_state["messages"])
                print("THREAD ID USED:", self.thread_id)

                try:
                    for event in graph.stream(
                        state,
                        config={"configurable": {"thread_id": self.thread_id}},
                    ):
                        for value in event.values():
                            for msg in value.get("messages", []):
                                if isinstance(msg, AIMessage):
                                    st.session_state["messages"].append(msg)
                                    with st.chat_message("assistant"):
                                        st.write(msg.content)
                except Exception as e:
                    st.error(
                        "The chatbot backend returned an authentication error.\n\n"
                        "This usually means the OpenAI / Groq API key or organisation "
                        "in your server is restricted or invalid. "
                        "Please update the key or organisation in your environment "
                        "and restart the app.\n\n"
                        f"Full error: {e}"
                    )

        # --------------------------------------------------------------
        # 2) CHATBOT WITH TAVILY SEARCH
        # --------------------------------------------------------------
        elif usecase == "Chatbot with tavily search":
            initial_state = {"messages": [HumanMessage(content=user_message)]}
            res = graph.invoke(initial_state)

            for message in res.get("messages", []):
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, ToolMessage):
                    with st.chat_message("assistant"):
                        st.markdown("**🧰 Tool Execution Started**")
                        st.write(message.content)
                        st.markdown("**✅ Tool Execution Completed**")
                elif isinstance(message, AIMessage) and getattr(
                    message, "content", None
                ):
                    with st.chat_message("assistant"):
                        st.write(message.content)
                else:
                    content = getattr(message, "content", str(message))
                    with st.chat_message("assistant"):
                        st.write(content)

        # --------------------------------------------------------------
        # 3) NEWS USECASE
        # --------------------------------------------------------------
        elif usecase == "News":
            news_type = st.session_state.get("NEWS_TYPE", "news")

            # Timeframe: Today / Weekly / Monthly
            timeframe_raw = (
                st.session_state.get("timeframe") or self.user_message or "Today"
            )
            timeframe = timeframe_raw.strip().capitalize()

            # Build JSON payload with timeframe + selected_date
            selected = st.session_state.get("selected_date")
            selected_iso = (
                selected.isoformat() if isinstance(selected, date) else None
            )
            payload = {"timeframe": timeframe.lower()}
            if selected_iso:
                payload["selected_date"] = selected_iso

            with st.spinner("Fetching and summarizing news... ⏳"):
                try:
                    _ = graph.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": json.dumps(payload),
                                }
                            ]
                        }
                    )
                except Exception as e:
                    st.warning(
                        "Graph invocation failed, using cached summaries if any.\n\n"
                        f"Details: {e}"
                    )

                # Load summary file based on timeframe
                filename = f"{timeframe.lower()}_summary.md".replace("today", "daily")
                news_path = os.path.join("News", filename)

                try:
                    with open(news_path, "r", encoding="utf-8", errors="ignore") as f:
                        markdown_content = f.read()
                except FileNotFoundError:
                    st.error(f"News not generated or file not found: {news_path}")
                    return
                except Exception as e:
                    st.error(f"An error occurred while reading news file: {e}")
                    return

                sections = parse_news_markdown_grouped(markdown_content)

                if not sections:
                    st.markdown(markdown_content, unsafe_allow_html=True)
                else:
                    render_news_sections(sections, news_type, timeframe)
