from src.LangGraph.state.state import State

class BasicChatbotNode:
    """
    Basic Chatbot logic Implementation
    """
    def __init__(self, model,news_type):
        self.llm = model
        self.news_type=news_type

    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a chatbot response.
        """
        # Keep existing messages
        # messages = state["messages"].copy()
        system_prompt = f"""
            You are an expert AI assistant specialized ONLY in the topic: **{self.news_type}**.

            ### Your Rules:
            1. You must answer **only questions related to {self.news_type}**.
            2. If the user asks anything **outside this topic**, reply with:
            "Please ask me questions related to {self.news_type}."
            3. If the user asks something partially related, clarify and bring the topic back.
            4. Keep answers short, factual, and focused on {self.news_type}.
            """
        # Get LLM reply
        last_user_msg = state["messages"][-1].content if state["messages"] else ""

        messages = [
            ("system", system_prompt),
            ("user", last_user_msg)
        ]
      
        response = self.llm.invoke(messages)

        # Convert to plain string if AIMessage object
        content = getattr(response, "content", str(response))

        # Append to chat history
        messages.append({"role": "assistant", "content": content})

        return {"messages": messages}