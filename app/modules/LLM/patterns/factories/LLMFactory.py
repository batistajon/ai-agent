from modules.LLM.domain.interfaces.ILLM import ILLM
from modules.LLM.domain.interfaces.ILLMFactory import ILLMFactory
from modules.LLM.patterns.strategies.DeepseekStrategy import DeepSeekStrategy


class LLMFactory(ILLMFactory):

    def __init__(self) -> None:
        self.llm_registry = {
          "deepseek": DeepSeekStrategy(),
        }

    def get_llm_strategy(self, llm_key: str) -> ILLM:
        llm_strategy = self.llm_registry[llm_key]
        return llm_strategy
