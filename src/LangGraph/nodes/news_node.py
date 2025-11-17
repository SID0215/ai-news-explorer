from datetime import datetime, timezone, date, timedelta
from collections import defaultdict
import json
import os
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree as ET

from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate

from src.LangGraph.state.state import State
from src.LangGraph.tools.search_tool import NewsDataSearch

# Optional HTTP client (Guardian + BBC)
try:
    import requests
except ImportError:
    requests = None


GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")


class NewsNode:
    """
    Node responsible for:
      1. Fetching raw articles (Guardian + BBC + Tavily + optional NewsData).
      2. Categorising & deduplicating them.
      3. Summarising into 60–150-word summaries.
      4. Saving markdown summaries per timeframe (daily/weekly/monthly).
    """

    def __init__(self, llm, news_type, tools):
        self.llm = llm
        self.news_type = (news_type or "news").lower().strip()
        self.tools = tools or []
        self.tavily = TavilyClient()
        self.guardian_key = GUARDIAN_API_KEY
        self.state = {}

    # ------------------------------------------------------------------
    # SMALL HELPERS
    # ------------------------------------------------------------------
    def _normalize_url(self, url: str) -> str:
        """Strip query/fragment and trailing slash for dedupe."""
        if not url:
            return ""
        try:
            p = urlparse(url.strip())
            p = p._replace(query="", fragment="")
            path = p.path.rstrip("/")
            p = p._replace(path=path)
            return urlunparse(p)
        except Exception:
            return url.strip().rstrip("/")

    def _category_match(self, item: dict) -> bool:
        """Rule-based filter to keep only items that match the chosen category."""
        cat = self.news_type
        if cat in ("news", "general"):
            return True  # no filter

        text = (
            (item.get("title") or "")
            + " "
            + (item.get("description") or "")
            + " "
            + (item.get("content") or "")
            + " "
            + (item.get("snippet") or "")
        ).lower()

        KEYWORDS = {
            "finance": [
                "stock",
                "stocks",
                "market",
                "markets",
                "earnings",
                "shares",
                "bond",
                "bonds",
                "profit",
                "loss",
                "revenue",
                "currency",
                "forex",
                "crypto",
                "bitcoin",
                "investor",
                "dividend",
            ],
            "business": [
                "company",
                "corporate",
                "merger",
                "acquisition",
                "startup",
                "earnings",
                "revenue",
                "profit",
                "loss",
                "ceo",
                "business",
            ],
            "sports": [
                "game",
                "match",
                "tournament",
                "league",
                "cup",
                "olympics",
                "world cup",
                "score",
                "goal",
                "win",
                "defeat",
                "team",
                "coach",
                "player",
            ],
            "movies": [
                "movie",
                "film",
                "box office",
                "trailer",
                "netflix",
                "hollywood",
                "bollywood",
                "series",
                "episode",
                "cinema",
                "streaming",
                "imdb",
                "rotten tomatoes",
            ],
            "tech": [
                "ai",
                "artificial intelligence",
                "chip",
                "semiconductor",
                "software",
                "app",
                "apps",
                "smartphone",
                "laptop",
                "cloud",
                "data center",
                "start-up",
                "startup",
                "cybersecurity",
                "robot",
                "tech",
                "technology",
            ],
        }

        kws = KEYWORDS.get(cat)
        if not kws:
            return True

        return any(k in text for k in kws)

    # ------------------------------------------------------------------
    # 1) GUARDIAN + BBC FETCHERS
    # ------------------------------------------------------------------
    def _fetch_guardian(self, days: int) -> list[dict]:
        """Fetch news from Guardian Content API."""
        if not (self.guardian_key and requests):
            return []

        today = date.today()
        from_date = (today - timedelta(days=days - 1)).isoformat()
        to_date = today.isoformat()

        section_map = {
            "news": None,
            "general": None,
            "finance": "business",
            "business": "business",
            "sports": "sport",
            "movies": "film",
            "tech": "technology",
        }
        section = section_map.get(self.news_type)

        params = {
            "api-key": self.guardian_key,
            "page-size": 50,
            "order-by": "newest",
            "from-date": from_date,
            "to-date": to_date,
            "show-fields": "headline,trailText,bodyText,thumbnail",
        }
        if section:
            params["section"] = section

        try:
            resp = requests.get(
                "https://content.guardianapis.com/search",
                params=params,
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        results = []
        for r in data.get("response", {}).get("results", []):
            fields = r.get("fields", {}) or {}
            title = fields.get("headline") or r.get("webTitle")
            desc = fields.get("trailText") or fields.get("bodyText") or ""
            url = r.get("webUrl")
            pub = r.get("webPublicationDate")

            if not (title and url):
                continue

            results.append(
                {
                    "title": title,
                    "description": desc,
                    "url": url,
                    "published_date": pub,
                    "source": "guardian",
                }
            )

        return results

    def _fetch_bbc(self, days: int) -> list[dict]:
        """Fetch news from BBC RSS feeds (free)."""
        if not requests:
            return []

        feed_map = {
            "news": "https://feeds.bbci.co.uk/news/rss.xml",
            "general": "https://feeds.bbci.co.uk/news/rss.xml",
            "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "finance": "https://feeds.bbci.co.uk/news/business/rss.xml",
            "sports": "https://feeds.bbci.co.uk/sport/rss.xml",
            "movies": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        }

        feed_url = feed_map.get(self.news_type)
        if not feed_url:
            return []

        try:
            resp = requests.get(feed_url, timeout=8)
            resp.raise_for_status()
            xml_bytes = resp.content
        except Exception:
            return []

        try:
            root = ET.fromstring(xml_bytes)
        except Exception:
            return []

        items = []
        today = datetime.now(timezone.utc).date()
        min_date = today - timedelta(days=days - 1)

        for item in root.findall(".//item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            title = title_el.text if title_el is not None else None
            desc = desc_el.text if desc_el is not None else ""
            url = link_el.text if link_el is not None else None
            pub_raw = pub_el.text if pub_el is not None else ""

            if not (title and url):
                continue

            # Rough date parsing; if it fails, assume today
            d = today
            if pub_raw:
                try:
                    # Example: Mon, 25 Mar 2024 10:23:00 GMT
                    dt = datetime.strptime(pub_raw[:25], "%a, %d %b %Y %H:%M:%S")
                    d = dt.date()
                except Exception:
                    pass

            if d < min_date or d > today:
                continue

            items.append(
                {
                    "title": title,
                    "description": desc,
                    "url": url,
                    "published_date": d.isoformat(),
                    "source": "bbc",
                }
            )

        return items

    # ------------------------------------------------------------------
    # 2) PRIMARY FETCH NEWS (GUARDIAN + BBC + TAVILY + OPTIONAL TOOL)
    # ------------------------------------------------------------------
    def fetch_news(self, state: dict) -> dict:
        """
        Fetch news based on frequency coming from the last user message.
        Frequency is "daily" / "weekly" / "monthly" (we normalise).
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

        # ---- Combine sources: Guardian + BBC + Tavily (+ tool) ----
        raw_results: list[dict] = []

        # Guardian
        guardian_items = self._fetch_guardian(days)
        raw_results.extend(guardian_items)

        # BBC
        bbc_items = self._fetch_bbc(days)
        raw_results.extend(bbc_items)

        # Tavily
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

        try:
            response = self.tavily.search(
                query=query,
                topic=tavily_topic,
                time_range=time_range,
                include_answer="none",
                max_results=35,
                days=days,
            )
            tavily_results = response.get("results", [])
        except Exception:
            tavily_results = []

        raw_results.extend(tavily_results)

        # Optional fallback: NewsDataSearch tool
        if not raw_results:
            news_tool = None
            for t in self.tools:
                if isinstance(t, NewsDataSearch):
                    news_tool = t
                    break

            if news_tool is not None:
                try:
                    tool_output = news_tool.run(f"latest {category} news")
                    tool_results = tool_output.get("results", [])
                    raw_results.extend(tool_results)
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # CLEAN, FILTER BY CATEGORY, DEDUP
        # ------------------------------------------------------------------
        today = date.today()
        clean_results = []
        seen_norm_urls = set()
        seen_article_keys = set()  # (domain, date, first 10 words of title)

        for item in raw_results:
            url = item.get("url") or item.get("link")
            title = item.get("title") or ""
            if not (url and title):
                continue

            # Category filter (finance only in finance, movies only in movies, etc.)
            if not self._category_match(item):
                continue

            # Date handling
            pub_raw = (
                item.get("published_date")
                or item.get("pubDate")
                or item.get("date")
                or ""
            )

            if pub_raw:
                try:
                    if "T" in pub_raw:
                        dt = datetime.fromisoformat(
                            pub_raw.replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                    else:
                        dt = datetime.fromisoformat(pub_raw)
                    d = dt.date()
                except Exception:
                    # already iso string from Guardian/BBC
                    try:
                        d = datetime.fromisoformat(pub_raw).date()
                    except Exception:
                        d = today
            else:
                d = today

            if d > today:
                # Skip future-dated articles to avoid "tomorrow's news"
                continue

            # Normalised URL dedupe
            norm_url = self._normalize_url(url)
            if norm_url in seen_norm_urls:
                continue

            # Also dedupe by (domain, date, first 10 words of title)
            parsed = urlparse(norm_url)
            domain = parsed.netloc.lower()
            first_words = " ".join(title.lower().split()[:10])
            article_key = (domain, d.isoformat(), first_words)
            if article_key in seen_article_keys:
                continue

            seen_norm_urls.add(norm_url)
            seen_article_keys.add(article_key)

            item["__pub_date_only"] = d.isoformat()
            item["__url"] = norm_url
            clean_results.append(item)

        self.state["news_data"] = clean_results
        state["news_data"] = clean_results
        return state

    # ------------------------------------------------------------------
    # 3) SUMMARISE ARTICLES (WITH SAFE FALLBACK)
    # ------------------------------------------------------------------
    def _build_articles_string(self, news_items: list) -> str:
        """
        Turn list of items into a compact text block
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

        Any LLM error is swallowed and we return [] so that the caller
        can fall back to non-LLM summaries.
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
        except Exception as e:
            print("[NewsNode] LLM summariser failed, using fallback:", repr(e))
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
        2. If it returns nothing (model didn't follow format or LLM blocked),
           fall back to using description/content directly (trimmed).
        """
        news_items = self.state.get("news_data", [])
        if not news_items:
            msg = "# No news found\n(No articles returned for this category and time range.)\n"
            self.state["summary"] = msg
            state["summary"] = msg
            return state

        # ---- 1) Try structured summariser ----
        articles_block = self._build_articles_string(news_items)
        structured = self._run_summariser(articles_block)

        # ---- 2) Fallback if LLM output could not be parsed ----
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
                lines.append(f"- **{title}**: {summary} [Read full story]({url})")
            lines.append("")

        summary_md = "\n".join(lines).strip()
        self.state["summary"] = summary_md
        state["summary"] = summary_md
        return state

    # ------------------------------------------------------------------
    # 4) SAVE SUMMARY TO MARKDOWN FILE
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
