from app.domain.interfaces.IFramework import IFramework


class FrameworkSingleton:
    """
    Concretion for FastAPI framework
    """
    def __init__(self, framework: IFramework):
        self._framework = framework

    def get_instance(self):
        return self._framework.get_instance()

    def load_routes(self, instance: object):
        return self._framework.load_routes(instance)

