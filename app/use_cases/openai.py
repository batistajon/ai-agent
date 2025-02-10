from logger import Logger
from langchain_openai import ChatOpenAI


logger = Logger().logger


class OpenAI:

    openai_llm: ChatOpenAI

    def _initializeChat():
        try:
            openai_llm = ChatOpenAI(
               model="gpt-4-1106-preview"
            )

            return openai_llm
            logger.info("ChatOllame was instantiated successfully!")

        except:
            logger.exception("Can't initialize ChatOllama.")
            raise
