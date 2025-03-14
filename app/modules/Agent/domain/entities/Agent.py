from app.domain.interfaces.IAgentAction import IAgentAction
from app.domain.interfaces.ILLM import ILLM

class Agent:
    def __init__(self, llm: ILLM, action: IAgentAction):
        self._llm = llm
        self._action = action

    def get_llm_intance(self):
        self._llm.get_instance()


    def run_action(self):
        self._action.run_action()



