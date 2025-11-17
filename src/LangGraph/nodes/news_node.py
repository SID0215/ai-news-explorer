from datetime import datetime, timezone, date, timedelta
from collections import defaultdict
import json
import os
from typing import List, Dict
from urllib.parse import urlparse, urlunparse

import requests
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate

from src.LangGraph.state.state import State
from src.LangGraph.tools.search_tool import NewsDataSearch


class NewsNode:
    """
    News node that:

      1. Fetches raw articles from multiple **free** sources:
         - Tavily (for latest / near-term)
         - BBC RSS feeds (latest)
         - The Guardian Content API (latest + archive)
         - GDELT Doc API (archive for selected dates)

      2. Summarises them into 60–150 word summaries.

      3. Writes markdown files for "daily", "weekly", "monthly" used by the UI.
    """

    def __init__(self, llm, news_type, tools):
        self.llm = llm
        self.news_type = (news_type or "news").lower().strip()
        self.tools = tools or []
        self.tavily = TavilyClient()
        self.state: Dict = {}
        self.guardian_key = os.getenv("GUARDIAN_API_KEY")

    # ------------------------------------------------------------------
    # URL NORMALISATION + DEDUPE
    # ------------------------------------------------------------------
    def _normalize_url(self, url: str) -> str:
        """
        Normalise URL so that UTM params / tracking do not create duplicates.
        """
        if not url:
            return ""
        try:
            url = url.strip()
            parsed = urlparse(url)
            cleaned = parsed._replace(query="", fragment="")
            scheme = cleaned.scheme.lower() or "https"
            netloc = cleaned.netloc.lower()
            cleaned = cleaned._replace(scheme=scheme, netloc=netloc)
            return urlunparse(cleaned)
        except Exception:
            return url.strip()

    def _dedupe_and_clamp_dates(self, items: List[Dict]) -> List[Dict]:
        """
        Remove duplicate URLs and clamp any future dates.

        Adds:
        - "__url"            : cleaned URL
        - "__pub_date_only"  : YYYY-MM-DD string
        """
        today = date.today()
        clean: List[Dict] = []
        seen: set[str] = set()

        for item in items:
            url = item.get("url") or item.get("link")
            if not url:
                continue

            norm = self._normalize_url(url)
            if not norm or norm in seen:
                continue
            seen.add(norm)

            pub_raw = (
                item.get("published_date")
                or item.get("pubDate")
                or item.get("date")
                or item.get("webPublicationDate")
                or item.get("seendate")
                or ""
            )

            if pub_raw:
                d = today
                try:
                    # ISO formats
                    if "T" in pub_raw:
                        dt = datetime.fromisoformat(
                            pub_raw.replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                        d = dt.date()
                    else:
                        dt = datetime.fromisoformat(pub_raw)
                        d = dt.date()
                except Exception:
                    # RSS-style: Mon, 17 Nov 2025 10:00:00 GMT
                    try:
                        dt = datetime.strptime(
                            pub_raw[:25], "%a, %d %b %Y %H:%M:%S"
                        )
                        d = dt.date()
                    except Exception:
                        d = today
            else:
                d = today

            if d > today:
                # Skip any future-dated articles
                continue

            item["__pub_date_only"] = d.isoformat()
            item["__url"] = url
            clean.append(item)

        return clean

    # ------------------------------------------------------------------
    # GUARDIAN (latest + archive)
    # ------------------------------------------------------------------
    def _fetch_guardian(
        self, start: date, end: date, category: str
    ) -> List[Dict]:
        if not self.guardian_key:
            return []

        url = "https://content.guardianapis.com/search"

        section_map = {
            "finance": "business",
            "business": "business",
            "sports": "sport",
            "movies": "film",
            "tech": "technology",
        }
        section = section_map.get(category)
        q = None
        if category in ("movies", "sports", "tech"):
            q = category

        params = {
            "api-key": self.guardian_key,
            "from-date": start.isoformat(),
            "to-date": end.isoformat(),
            "page-size": 50,
            "order-by": "newest",
            "show-fields": "trailText,bodyText",
        }
        if section:
            params["section"] = section
        if q:
            params["q"] = q

        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        results: List[Dict] = []
        for r in data.get("response", {}).get("results", []):
            web_url = r.get("webUrl")
            if not web_url:
                continue
            fields = r.get("fields", {}) or {}
            results.append(
                {
                    "title": r.get("webTitle"),
                    "description": fields.get("trailText")
                    or fields.get("bodyText")
                    or "",
                    "url": web_url,
                    "published_date": r.get("webPublicationDate", ""),
                    "source": "guardian",
                }
            )
        return results

    # ------------------------------------------------------------------
    # BBC RSS (latest headlines per category – free)
    # ------------------------------------------------------------------
    def _fetch_bbc(self, category: str) -> List[Dict]:
        feed_map = {
            "news": "https://feeds.bbci.co.uk/news/rss.xml",
            "general": "https://feeds.bbci.co.uk/news/rss.xml",
            "finance": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "sports": "https://feeds.bbci.co.uk/sport/rss.xml",
            "movies": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        }
        feed_url = feed_map.get(category, feed_map["news"])

        try:
            resp = requests.get(feed_url, timeout=8)
            resp.raise_for_status()
            import xml.etree.ElementTree as ET

            root = ET.fromstring(resp.content)
        except Exception:
            return []

        items: List[Dict] = []
        for node in root.findall(".//item"):
            title = node.findtext("title") or ""
            desc = node.findtext("description") or ""
            link = node.findtext("link") or ""
            pub = node.findtext("pubDate") or ""
            if not link:
                continue
            items.append(
                {
                    "title": title,
                    "description": desc,
                    "url": link,
                    "published_date": pub,
                    "source": "bbc",
                }
            )
        return items

    # ------------------------------------------------------------------
    # GDELT DOC API (archive)
    # ------------------------------------------------------------------
    def _fetch_gdelt(
        self, start: date, end: date, category: str
    ) -> List[Dict]:
        """
        Use GDELT doc API for archive ranges (for any selected date
        that is strictly in the past).
        """
        base = "http://api.gdeltproject.org/api/v2/doc/doc"

        query_map = {
            "finance": "finance OR stock OR market",
            "business": "business OR company OR earnings",
            "sports": "sports OR football OR cricket OR soccer OR tennis",
            "movies": "movie OR film OR cinema OR hollywood OR bollywood",
            "tech": "technology OR AI OR software OR gadgets OR startups",
            "general": "",
            "news": "",
        }
        extra = query_map.get(category, "")
        query = "news"
        if extra:
            query = f"news {extra}"

        start_dt = start.strftime("%Y%m%d000000")
        end_dt = end.strftime("%Y%m%d235959")

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": 50,
            "sort": "Date",
            "format": "json",
            "startdatetime": start_dt,
            "enddatetime": end_dt,
        }

        try:
            resp = requests.get(base, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        items: List[Dict] = []
        for art in data.get("articles", []):
            url = art.get("url")
            if not url:
                continue
            items.append(
                {
                    "title": art.get("title"),
                    "description": art.get("sourceurl") or "",
                    "url": url,
                    "published_date": art.get("seendate"),
                    "source": "gdelt",
                }
            )
        return items

    # ------------------------------------------------------------------
    # 1) FETCH RAW NEWS
    # ------------------------------------------------------------------
    def fetch_news(self, state: dict) -> dict:
        """
        Fetch news based on timeframe + selected_date from the UI.

        Last user message is either:
          - JSON string: {"timeframe": "...", "selected_date": "YYYY-MM-DD"}
          - plain string: "today"/"weekly"/"monthly"
        """
        last_msg = state["messages"][-1]["content"]

        if isinstance(last_msg, str):
            try:
                payload = json.loads(last_msg)
            except json.JSONDecodeError:
                payload = {"timeframe": last_msg}
        elif isinstance(last_msg, dict):
            payload = last_msg
        else:
            payload = {}

        frequency = str(payload.get("timeframe", "today")).lower()
        selected_date_str = payload.get("selected_date")

        # Normalise timeframe
        if frequency in ("today", "daily"):
            frequency = "daily"
        elif frequency.startswith("week"):
            frequency = "weekly"
        elif frequency.startswith("month"):
            frequency = "monthly"
        else:
            frequency = "daily"

        # Anchor date (never in the future)
        today = date.today()
        if selected_date_str:
            try:
                anchor = datetime.fromisoformat(selected_date_str).date()
            except Exception:
                anchor = today
        else:
            anchor = today
        if anchor > today:
            anchor = today

        # Compute range for Guardian / GDELT
        if frequency == "daily":
            start_date = end_date = anchor
        elif frequency == "weekly":
            start_date = anchor - timedelta(days=6)
            end_date = anchor
        else:  # monthly
            start_date = anchor - timedelta(days=29)
            end_date = anchor

        self.state["frequency"] = frequency
        self.state["selected_date"] = anchor.isoformat()

        category = self.news_type
        all_items: List[Dict] = []

        # ----------------------------------------------------------
        # Tavily + BBC – only if the range touches *today*
        # (Tavily is good for recent, not deep archives)
        # ----------------------------------------------------------
        if end_date >= today:
            time_range_map = {"daily": "day", "weekly": "week", "monthly": "month"}
            days_map = {"daily": 1, "weekly": 7, "monthly": 30}

            time_range = time_range_map.get(frequency, "day")
            days = days_map.get(frequency, 1)

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
            query = f"Latest {category} news – {query_suffix}"

            try:
                tavily_resp = self.tavily.search(
                    query=query,
                    topic=tavily_topic,
                    time_range=time_range,
                    include_answer="none",
                    max_results=35,
                    days=days,
                )
                tavily_results = tavily_resp.get("results", [])
            except Exception:
                tavily_results = []

            all_items.extend(tavily_results)

            # BBC headlines (always latest)
            all_items.extend(self._fetch_bbc(category))

        # Guardian (works for both latest + archive)
        all_items.extend(self._fetch_guardian(start_date, end_date, category))

        # GDELT – only if anchor is in the past
        if anchor < today:
            all_items.extend(self._fetch_gdelt(start_date, end_date, category))

        # Optional fallback: NewsDataSearch tool
        if not all_items:
            news_tool = next(
                (t for t in self.tools if isinstance(t, NewsDataSearch)), None
            )
            if news_tool is not None:
                try:
                    tool_output = news_tool.run(f"latest {category} news")
                    all_items.extend(tool_output.get("results", []))
                except Exception:
                    pass

        # Final cleaning + de-dupe
        clean_results = self._dedupe_and_clamp_dates(all_items)
        self.state["news_data"] = clean_results
        state["news_data"] = clean_results
        return state

    # ------------------------------------------------------------------
    # 2) SUMMARISE ARTICLES  (STRICT, LOW HALLUCINATION)
    # ------------------------------------------------------------------
    def _build_articles_string(self, news_items: List[Dict]) -> str:
        """
        Turn article list into a compact text block for the LLM.
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

            # single-line representation to avoid parsing issues
            blocks.append(
                f"ID: {idx} | DATE: {pub} | TITLE: {title} | TEXT: {desc} | URL: {url}"
            )
        return "\n".join(blocks)

    def _run_summariser(self, articles_block: str) -> List[Dict]:
        """
        Call LLM and ask for strict structured summaries.

        Output format per line:
            DATE || HEADLINE || SUMMARY || URL
        """
        if not articles_block:
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
            raw = getattr(response, "content", str(response))
        except Exception:
            # If LLM call fails (e.g., org restricted), we fall back later.
            return []

        summaries: List[Dict] = []
        seen_urls: set[str] = set()

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
        Summarise fetched news into markdown understood by the UI.
        """
        news_items = self.state.get("news_data", [])
        if not news_items:
            msg = "# No news found\n(No articles returned for this category and time range.)\n"
            self.state["summary"] = msg
            state["summary"] = msg
            return state

        # 1) Try strict LLM summariser
        articles_block = self._build_articles_string(news_items)
        structured = self._run_summariser(articles_block)

        # 2) Fallback using descriptions directly
        if not structured:
            fallback: List[Dict] = []
            seen_urls: set[str] = set()

            for item in news_items:
                url = item.get("__url") or item.get("url") or item.get("link")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                d = item.get("__pub_date_only") or date.today().isoformat()
                title = item.get("title") or "News"

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
                    summary = " ".join(words[:150])

                fallback.append(
                    {
                        "date": d,
                        "title": title,
                        "summary": summary,
                        "url": url,
                    }
                )

            structured = fallback

        # 3) Group by date → markdown
        grouped: dict[str, List[Dict]] = defaultdict(list)
        for item in structured:
            d = item.get("date") or date.today().isoformat()
            grouped[d].append(item)

        lines: List[str] = []
        for d in sorted(grouped.keys(), reverse=True):
            lines.append(f"### {d}")
            for art in grouped[d]:
                title = art["title"]
                summary = art["summary"]
                url = art["url"]
                lines.append(f"- **{title}**: {summary} [Read full story]({url})")
            lines.append("")

        summary_md = "\n".join(lines).strip()
        self.state["summary"] = summary_md
        state["summary"] = summary_md
        return state

    # ------------------------------------------------------------------
    # 3) SAVE SUMMARY FILE
    # ------------------------------------------------------------------
    def save_result(self, state: State, config=None):
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

        os.makedirs("./News", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {heading}\n\n")
            f.write(summary)

        self.state["filename"] = filename
        return self.state
