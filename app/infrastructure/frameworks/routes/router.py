from fastapi import APIRouter, Request, status
from starlette.responses import JSONResponse

from modules.LLM.patterns.factories.LLMFactory import LLMFactory


router = APIRouter()


@router.get("/")
def index() -> JSONResponse:
    response = JSONResponse({
        "error": False,
        "message": "Agent API is ready."
    }, status_code=200)

    return response


@router.post("/agent/train/pdf")
def training():
    response = JSONResponse({
        "error": False,
        "message": "training by PDF"
    }, status_code=201)

    return response


@router.post("/agent/ask")
def askRAG():
    llm_factory = LLMFactory()
    llm_strategy = llm_factory.get_llm_strategy(llm_key="deepseek")
    llm = llm_strategy.get_instance()

    response = JSONResponse({
        "error": False,
        "message": llm
    }, status_code=200)

    return response

