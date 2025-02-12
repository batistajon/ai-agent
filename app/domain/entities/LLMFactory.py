from app.domain.interfaces.ILLM import ILLM
from app.domain.interfaces.ILLMFactory import ILLMFactory
from app.use_cases.deepseek_strategy import DeepSeekStrategy


class LLMFactory(ILLMFactory):

    def __init__(self) -> None:
        self.llm_registry = {
          "deepseek": DeepSeekStrategy(),
        }

    def get_llm_strategy(self, llm_key: str) -> ILLM:
        llm_strategy = self.llm_registry[llm_key]
        return llm_strategy
