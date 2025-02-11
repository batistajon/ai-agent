from abc import ABC, abstractmethod


class IFramework(ABC):
    """
    Encapsulates Framework rules
    """
    @abstractmethod
    def get_instance(self) -> object:
        """
        Gets framework instance
        """
        pass

    @abstractmethod
    def load_routes(self, instance: object):
        """
        Loads all routes for the give framework
        """
        pass

