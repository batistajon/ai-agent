from logger import Logger
from langchain_community.chat_models import ChatOllama


logger = Logger().logger


class Ollama:

    ollama_llm: ChatOllama

    def _initializeChat():
        try:
            ollama_llm = ChatOllama(
                model="llama3",
                base_url="http://localhost:11434"
            )

            return ollama_llm

        except:
            logger.error("Can't initialize ChatOllama.")
            raise
