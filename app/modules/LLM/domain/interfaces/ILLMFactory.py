from abc import ABC, abstractmethod
from modules.LLM.domain.interfaces.ILLM import ILLM


class ILLMFactory(ABC):
    """
    Encapsulates LLM creation rules
    """

    @abstractmethod
    def get_llm_strategy(self, llm_key: str) -> ILLM:
        """
        Gets the LLM instance according to the registred LLMs
        """
        pass



