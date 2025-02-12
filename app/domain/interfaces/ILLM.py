from abc import ABC, abstractmethod


class ILLM(ABC):
    """
    Encapsulates LLM rules
    """


    @abstractmethod
    def get_instance(self) -> object:
        """
        Gets the LLM instance according to the registred LLMs
        """
        pass


