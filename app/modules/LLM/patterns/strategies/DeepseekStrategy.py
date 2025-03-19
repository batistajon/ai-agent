from modules.LLM.domain.interfaces.ILLM import ILLM
from openai import OpenAI


class DeepSeekStrategy(ILLM):
    def get_instance(self) -> object:
        return {"status": "it's working!"}
