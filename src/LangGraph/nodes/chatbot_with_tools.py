from src.LangGraph.state.state import State

class ChatBotToolNode:
    def __init__(self,model):
        self.llm=model 
    
    # with no tools
    def process(self,state:State)->dict:
        """
        Processes the input state and generate a response with tool integration
        """
        # just tkaing las t message from user
        user_input = state["messages"][-1] if state['messages'] else ""
        llm_response = self.llm.invoke({"role":"user","content":user_input})

        # Simulate tool-specific logic
        tools_response=f"Tool integration for:'{user_input}'"

        return {"messages":[tools_response,llm_response]}
    
    def create_chatbot(self,tools):
            '''
                Returns a chatbot node function
            '''
            llm_with_tools = self.llm.bind_tools(tools)

            def chatbot_node(state:State):
                """
                   Chatbot Logic for processing the input state and returning a response 
                """
                print(state)
                # return {"messages":[llm_with_tools.invoke(state["messages"])]}
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": state["messages"] + [response]}
             
            return chatbot_node