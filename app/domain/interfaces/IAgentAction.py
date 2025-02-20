from abc import ABC, abstractmethod

class IAgentAction(ABC):
    @abstractmethod
    def run_action(self) -> None:
        """
        Defines Quack behavior.
        """
        pass

