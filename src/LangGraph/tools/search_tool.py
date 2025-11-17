from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain.tools import BaseTool
from newsdataapi import NewsDataApiClient
from pydantic import PrivateAttr
from typing import Any, Dict
from dotenv import load_dotenv
import os
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

NEWS_DATA_API_KEY = os.getenv("NEWS_DATA_API_KEY")

class NewsDataSearch(BaseTool):
    name: str = "newsdata_search"
    description: str = (
        "Search global news articles using the NewsData.io API. "
        "Input must be a text query."
    )

    api_key: str
    _client: Any = PrivateAttr()

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self._client = NewsDataApiClient(apikey=api_key)

    def _run(self, query: str, days: int = None) -> Dict:
        """
        Run NewsData.io search with optional time filtering.
        
        """
        
        
       
        response = self._client.news_api(
            q=query,
            language="en",
            country="us"
        )
        return response

    async def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")

def get_tools():
    """
    Return the list of tools to be used in the chatbot
    """
    tools=[TavilySearchResults(api_key=TAVILY_API_KEY,max_results=2),
           NewsDataSearch(api_key=NEWS_DATA_API_KEY)
           ]
    return tools

def create_tool_node(tools):
    """
    creates and returns a tool node for the graph
    """
    return ToolNode(tools=tools)