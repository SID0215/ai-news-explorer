from langgraph.graph import StateGraph
from src.LangGraph.state.state import State
from langgraph.graph import START,END
from src.LangGraph.nodes.basic_chatbot_node import BasicChatbotNode
from src.LangGraph.tools.search_tool import get_tools,create_tool_node
from langgraph.prebuilt import ToolNode,tools_condition
from src.LangGraph.nodes.chatbot_with_tools import ChatBotToolNode

from src.LangGraph.nodes.news_node import NewsNode
class GraphBuilder:
    def __init__(self,model,news_type):
        self.llm=model
        self.news_type=news_type
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        self.basic_chatbot_node=BasicChatbotNode(self.llm,self.news_type)

        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)

    def chatbot_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """

        ##Defining the tool
        tools=get_tools()
       
        tavily_tools = [
            t for t in tools 
            if "tavily" in t.name.lower()   # match substrings safely
        ]
        tool_node = create_tool_node(tavily_tools)
        ## Define LLM
        llm=self.llm

        # Define chatbot node
        object_chatbot_with_tools=ChatBotToolNode(llm)
        chatbot_node=object_chatbot_with_tools.create_chatbot(tools=tools)
        # Add Node
        self.graph_builder.add_node("chatbot",chatbot_node)
        self.graph_builder.add_node("tools",tool_node)
        
        # Define conditional edge and direct edge
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        # self.graph_builder.add_edge("tools","chatbot")
        self.graph_builder.add_edge("tools", END)  # FIXED — no recursion

    
    def news_builder_graph(self):
        tools = get_tools()            # ← LIST of tools
        tool_node = create_tool_node(tools)  # ← ToolNode (NOT passed to NewsNode)

        # pass only tools list
        news_node = NewsNode(self.llm, self.news_type, tools)

        self.graph_builder.add_node("fetch_news", news_node.fetch_news)
        self.graph_builder.add_node("summarize_news", news_node.summarize_news)
        self.graph_builder.add_node("save_results", news_node.save_result)

        self.graph_builder.set_entry_point("fetch_news")
        self.graph_builder.add_edge("fetch_news", "summarize_news")
        self.graph_builder.add_edge("summarize_news", "save_results")
        self.graph_builder.add_edge("save_results", END)

    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        if usecase == "Chatbot with tavily search":
            self.chatbot_tools_build_graph()
        if usecase == "News":
            self.news_builder_graph()
        return self.graph_builder.compile()
