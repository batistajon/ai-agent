from ollama import Ollama
from openai import OpenAI


class LoadLLM:
    llm_map = {
        "openai": OpenAI,
        "ollama": Ollama,
    }

    def __init__(self, llm: str):
        if llm in self.llm_map:
            return self.llm_map.get(llm)

        return False
