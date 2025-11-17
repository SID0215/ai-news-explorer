from datetime import datetime, timezone, date
from collections import defaultdict
import json
import re

from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate

from src.LangGraph.state.state import State
from src.LangGraph.tools.search_tool import NewsDataSearch


class NewsNode:
    """
    Node responsible for:
      1. Fetching raw articles (Tavily + optional NewsData fallback).
      2. Summarising them into ~60–150-word summaries.
      3. Saving markdown summaries per timeframe (daily/weekly/monthly).

    If the LLM is blocked (organization_restricted or any 400/401 error),
    we FALL BACK to using the original article text (description/content),
    so the app still works and does NOT hallucinate.

    Also adds a deterministic category filter so:
      - sports → sports-related
      - movies → film/TV/entertainment
      - tech → technology
      - finance/business → financial / business
    """

    def __init__(self, llm, news_type, tools):
        # llm CAN be None (if org is restricted). We handle that in summarize.
        self.llm = llm
        self.news_type = (news_type or "news").lower().strip()
        self.tools = tools or []
        self.tavily = TavilyClient()
        self.state = {}

    # ------------------------------------------------------------------
    # 1) FETCH RAW NEWS
    # ------------------------------------------------------------------
    def fetch_news(self, state: dict) -> dict:
        """
        Fetch news based on frequency coming from the last user message.
        The message is either:
          - "today" / "daily" / "weekly" / "monthly"
          - or a JSON string in the future (we keep it backwards-compatible).
        """

        last_msg = state["messages"][-1]["content"]
        if isinstance(last_msg, str):
            try:
                payload = json.loads(last_msg)
                frequency = payload.get("timeframe", "today")
            except json.JSONDecodeError:
                frequency = last_msg.strip().lower()
        elif isinstance(last_msg, dict):
            frequency = last_msg.get("timeframe", "today")
        else:
            frequency = "today"

        # Normalise
        if frequency in ("today", "daily"):
            frequency = "daily"
        elif frequency.startswith("week"):
            frequency = "weekly"
        elif frequency.startswith("month"):
            frequency = "monthly"

        self.state["frequency"] = frequency

        time_range_map = {"daily": "day", "weekly": "week", "monthly": "month"}
        days_map = {"daily": 1, "weekly": 7, "monthly": 30}

        time_range = time_range_map.get(frequency, "day")
        days = days_map.get(frequency, 1)

        # Map our UI categories to Tavily topics + query flavours
        category = self.news_type
        CATEGORY_CONFIG = {
            "news": (
                "news",
                "breaking news headlines from BBC, The Guardian, AP and Reuters",
            ),
            "general": (
                "news",
                "top general stories from BBC, The Guardian, AP and Reuters",
            ),
            "finance": (
                "finance",
                "finance and markets news from Reuters, Bloomberg, WSJ and FT",
            ),
            "business": (
                "finance",
                "business and company news from FT, Bloomberg, WSJ and Reuters",
            ),
            "sports": (
                "news",
                "sports headlines, scores and match reports from ESPN and BBC Sport",
            ),
            "movies": (
                "news",
                "movies and entertainment news from Variety, Hollywood Reporter and IMDB news",
            ),
            "tech": (
                "news",
                "technology news about AI, software, gadgets and startups from The Verge, Wired and TechCrunch",
            ),
        }

        tavily_topic, query_suffix = CATEGORY_CONFIG.get(
            category, ("news", "breaking news")
        )

        query = f"Latest {category} news today – {query_suffix}"

        # --- Primary: Tavily ---
        try:
            response = self.tavily.search(
                query=query,
                topic=tavily_topic,
                time_range=time_range,
                include_answer="none",
                max_results=40,
                days=days,
            )
            results = response.get("results", [])
        except Exception as e:
            print(f"[NewsNode] Tavily fetch failed: {e}")
            results = []

        # --- Optional fallback: NewsDataSearch tool ---
        if not results:
            news_tool = None
            for t in self.tools:
                if isinstance(t, NewsDataSearch):
                    news_tool = t
                    break

            if news_tool is not None:
                try:
                    tool_output = news_tool.run(f"latest {category} news")
                    results = tool_output.get("results", [])
                except Exception as e:
                    print(f"[NewsNode] NewsData fetch failed: {e}")
                    results = []

        # Clean / clamp dates; deduplicate by URL
        today = date.today()
        clean_results = []
        seen_urls = set()

        for item in results:
            url = item.get("url") or item.get("link")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = item.get("title") or ""
            desc = (
                item.get("description")
                or item.get("content")
                or item.get("snippet")
                or ""
            )
            text_for_cat = f"{title} {desc}"

            # Try to get a date; clamp to today if future
            pub_raw = (
                item.get("published_date")
                or item.get("pubDate")
                or item.get("date")
                or ""
            )

            if pub_raw:
                try:
                    # Handle common ISO and RSS formats
                    if "T" in pub_raw:
                        dt = datetime.fromisoformat(
                            pub_raw.replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                    else:
                        dt = datetime.fromisoformat(pub_raw)
                    d = dt.date()
                except Exception:
                    d = today
            else:
                d = today

            if d > today:
                # Skip future-dated articles to avoid "tomorrow's news"
                continue

            item["__pub_date_only"] = d.isoformat()
            item["__url"] = url
            item["__cat_text"] = text_for_cat
            clean_results.append(item)

        # Apply deterministic category filter to reduce cross-noise
        filtered_results = self._filter_by_category(clean_results, category)

        self.state["news_data"] = filtered_results
        state["news_data"] = filtered_results
        return state

    # ------------------------------------------------------------------
    # CATEGORY FILTER
    # ------------------------------------------------------------------
    def _filter_by_category(self, items, category: str):
        """
        Simple keyword-based post-filtering to reduce wrong-category articles.

        We:
          - keep items matching category-specific keywords
          - if NOTHING matches, fall back to original list (so UI isn't empty)
        """
        if not items:
            return items

        category = (category or "").lower()
        cat_items = []

        def contains_any(text: str, keywords):
            text_low = text.lower()
            return any(kw in text_low for kw in keywords)

        for it in items:
            txt = it.get("__cat_text", "")

            # Sports
            if category == "sports":
                sports_kw = [
                    "match",
                    "game",
                    "tournament",
                    "league",
                    "cup",
                    "goal",
                    "score",
                    "scored",
                    "team",
                    "coach",
                    "olympics",
                    "cricket",
                    "football",
                    "soccer",
                    "nba",
                    "nfl",
                    "fifa",
                    "tennis",
                    "grand slam",
                ]
                if contains_any(txt, sports_kw):
                    cat_items.append(it)
                continue

            # Movies / entertainment
            if category == "movies":
                movie_kw = [
                    "film",
                    "movie",
                    "cinema",
                    "box office",
                    "trailer",
                    "series",
                    "show",
                    "season",
                    "episode",
                    "netflix",
                    "disney+",
                    "prime video",
                    "imdb",
                    "rotten tomatoes",
                    "hollywood",
                    "bollywood",
                    "cast",
                    "director",
                    "actor",
                    "actress",
                    "oscar",
                    "award",
                ]
                if contains_any(txt, movie_kw):
                    cat_items.append(it)
                continue

            # Tech
            if category == "tech":
                tech_kw = [
                    "ai",
                    "artificial intelligence",
                    "machine learning",
                    "software",
                    "app",
                    "startup",
                    "cloud",
                    "data center",
                    "semiconductor",
                    "chip",
                    "processor",
                    "nvidia",
                    "intel",
                    "amd",
                    "microsoft",
                    "google",
                    "alphabet",
                    "meta",
                    "facebook",
                    "apple",
                    "iphone",
                    "android",
                    "cybersecurity",
                    "hack",
                    "breach",
                    "blockchain",
                    "crypto",
                ]
                if contains_any(txt, tech_kw):
                    cat_items.append(it)
                continue

            # Finance
            if category == "finance":
                fin_kw = [
                    "market",
                    "stocks",
                    "stock",
                    "equity",
                    "shares",
                    "bond",
                    "bonds",
                    "treasury",
                    "interest rate",
                    "fed",
                    "inflation",
                    "recession",
                    "earnings",
                    "revenue",
                    "profit",
                    "loss",
                    "bank",
                    "loan",
                    "credit",
                    "fund",
                    "investment",
                    "investor",
                ]
                if contains_any(txt, fin_kw):
                    cat_items.append(it)
                continue

            # Business
            if category == "business":
                biz_kw = [
                    "company",
                    "corporate",
                    "merger",
                    "acquisition",
                    "startup",
                    "layoff",
                    "restructuring",
                    "revenue",
                    "earnings",
                    "profit",
                    "loss",
                    "ceo",
                    "founder",
                    "ipo",
                    "shareholders",
                    "board of directors",
                ]
                if contains_any(txt, biz_kw):
                    cat_items.append(it)
                continue

            # news / general → keep everything (already deduped & time-filtered)
            if category in {"news", "general"}:
                cat_items.append(it)
                continue

        # If our filter killed everything, return original set
        if cat_items:
            return cat_items
        return items

    # ------------------------------------------------------------------
    # 2) SUMMARISE ARTICLES (STRONG GUARDRAILS)
    # ------------------------------------------------------------------
    def _build_articles_string(self, news_items: list) -> str:
        """
        Turn list of Tavily/NewsData items into a compact text block
        that the LLM can safely summarise.
        """
        blocks = []
        for idx, item in enumerate(news_items, start=1):
            title = item.get("title") or ""
            desc = (
                item.get("description")
                or item.get("content")
                or item.get("snippet")
                or ""
            )
            url = item.get("__url") or item.get("url") or item.get("link")
            pub = item.get("__pub_date_only") or ""
            if not title or not url:
                continue

            text = f"ID: {idx}\nDATE: {pub}\nTITLE: {title}\nTEXT: {desc}\nURL: {url}"
            blocks.append(text)

        return "\n\n".join(blocks)

    def _run_summariser(self, articles_block: str) -> list[dict]:
        """
        Call LLM once and ask for structured, non-hallucinated summaries.

        Output format (one line per article):

            DATE || HEADLINE || SUMMARY || URL

        where SUMMARY is 60–150 words.

        If the LLM call fails (e.g., organization_restricted),
        we CATCH the exception and return [] so we can fall back to
        using original text (no crash).
        """
        if not articles_block or self.llm is None:
            # If llm is None, skip LLM summarisation entirely
            return []

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a STRICT news summarisation engine.

You MUST:
- Only use information that appears in the provided articles.
- Never invent events, numbers, quotes, people or dates.
- Never create imaginary news or modify the tone.
- Skip index pages, category pages, or pages without real article text.

For each valid article, output ONE line exactly in this format:

DATE || HEADLINE || SUMMARY || URL

Rules:
- DATE: ISO format YYYY-MM-DD (use the DATE field provided).
- HEADLINE: 6–14 words, no newlines.
- SUMMARY: 60–150 words, 2–4 sentences, plain English.
- URL: the original article URL.
- If the article should be ignored (listing / duplicate / junk), output nothing.
- Do NOT add bullet points, explanations, or any extra text.
""",
                ),
                ("user", "Here are the articles:\n\n{articles}"),
            ]
        )

        try:
            response = self.llm.invoke(prompt.format(articles=articles_block))
        except Exception as e:
            # THIS IS WHERE YOUR 400 / organization_restricted ERROR WAS COMING FROM.
            # Now we catch it and fall back to “no structured summary”.
            print(f"[NewsNode] LLM summariser failed: {e}")
            return []

        raw = getattr(response, "content", str(response))

        summaries: list[dict] = []
        seen_urls = set()

        for line in raw.splitlines():
            line = line.strip()
            if not line or "||" not in line:
                continue

            parts = [p.strip() for p in line.split("||")]
            if len(parts) < 4:
                continue

            date_str, headline, summary, url = parts[:4]
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            summaries.append(
                {
                    "date": date_str or "",
                    "title": headline,
                    "summary": summary,
                    "url": url,
                }
            )

        return summaries

    def summarize_news(self, state: dict) -> dict:
        """
        Summarise the fetched news items into markdown that the UI understands.

        1. Try strict LLM summariser (_run_summariser).
        2. If it returns nothing (model didn't follow format OR LLM failed),
           fall back to using description/content directly (trimmed).
        """
        news_items = self.state.get("news_data", [])
        if not news_items:
            msg = "# No news found\n(No articles returned for this category and time range.)\n"
            self.state["summary"] = msg
            state["summary"] = msg
            return state

        # ---- 1) Try strict structured summariser ----
        articles_block = self._build_articles_string(news_items)
        structured = self._run_summariser(articles_block)

        # ---- 2) Fallback if LLM output could not be parsed OR failed ----
        if not structured:
            fallback = []
            seen_urls = set()

            for item in news_items:
                url = item.get("__url") or item.get("url") or item.get("link")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                d = item.get("__pub_date_only") or date.today().isoformat()
                title = item.get("title") or "News"

                # Use whatever real text we have from the source
                text = (
                    item.get("description")
                    or item.get("content")
                    or item.get("snippet")
                    or ""
                )

                if not text:
                    summary = (
                        "Source did not provide article text. "
                        "Open the full story to read more."
                    )
                else:
                    words = text.split()
                    summary = " ".join(words[:150])  # up to ~150 words

                fallback.append(
                    {
                        "date": d,
                        "title": title,
                        "summary": summary,
                        "url": url,
                    }
                )

            structured = fallback

        # ---- 3) Group by date and build markdown ----
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in structured:
            d = item.get("date") or date.today().isoformat()
            grouped[d].append(item)

        lines = []
        # newest dates first
        for d in sorted(grouped.keys(), reverse=True):
            lines.append(f"### {d}")
            for art in grouped[d]:
                title = art["title"]
                summary = art["summary"]
                url = art["url"]
                # Headline in bold + summary + link
                lines.append(f"- **{title}**: {summary} [Read full story]({url})")
            lines.append("")

        summary_md = "\n".join(lines).strip()
        self.state["summary"] = summary_md
        state["summary"] = summary_md
        return state

    # ------------------------------------------------------------------
    # 3) SAVE SUMMARY TO MARKDOWN FILE
    # ------------------------------------------------------------------
    def save_result(self, state, config=None):
        frequency = self.state.get("frequency", "daily")
        summary = self.state.get("summary", "")

        if not summary:
            self.state["filename"] = None
            return self.state

        filename = f"./News/{frequency}_summary.md"
        heading = {
            "daily": "Today News Summary",
            "weekly": "Weekly News Summary",
            "monthly": "Monthly News Summary",
        }.get(frequency, "Daily News Summary")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {heading}\n\n")
            f.write(summary)

        self.state["filename"] = filename
        return self.state
