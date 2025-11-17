import streamlit as st

from src.LangGraph.ui.streamlitui.loadui import LoadStreamLitUI
from src.LangGraph.llms.groqllm import GroqLLM
from src.LangGraph.graph.graph_builder import GraphBuilder
from src.LangGraph.ui.streamlitui.display_results import DisplayResultStreamlit


def load_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.
    """

    ui = LoadStreamLitUI()
    user_input = ui.load_streamlit()

    if not user_input:
        st.error("Error: Failed to load user input from ui")
        return

    # Always have a news_type string, even if the radio isn't shown
    news_type = user_input.get("NEWS_TYPE", st.session_state.get("NEWS_TYPE", "news"))
    st.session_state["NEWS_TYPE"] = news_type

    if (
        st.session_state.IsFetchButtonClicked
        or st.session_state.IsFetchAIButtonClicked
    ):
        user_message = st.session_state.timeframe
    else:
        user_message = st.chat_input("Enter the messsage:")

    print(news_type)
    print(st.session_state["thread_id"])

    if user_message:
        try:
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model()
            thread_id = st.session_state["thread_id"]
            if not model:
                st.error("Error: LLM model could not be initialized")
                return

            # Initialize and set up the graph based on use case
            usecase = user_input["USE_CASE_OPTIONS"]
            st.write("Selected usecase:", usecase)

            if not usecase:
                st.error("Error: No use case selected.")
                return

            graph_builder = GraphBuilder(model, news_type)
            try:
                graph = graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(
                    usecase, graph, user_message, thread_id
                ).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph set up failed- {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph set up failed- {e}")
            return
