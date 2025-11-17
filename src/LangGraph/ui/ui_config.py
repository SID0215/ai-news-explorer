from configparser import ConfigParser

class Config:
    def get_title(self):
        # New heading at the top of the Streamlit app
        return "AI News Explorer: Smart Daily, Weekly & Monthly Briefings"
    
    def __init__(self,config_file_path="./src/langGraph/ui/ui_config.ini"):
        self.config=ConfigParser()
        self.config.read(config_file_path)

    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")
    
    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")
    
    def get_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")