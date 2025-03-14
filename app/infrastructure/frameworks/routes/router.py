from fastapi import APIRouter, Request, status
from starlette.responses import JSONResponse

from app.patterns.factories import LLMFactory


router = APIRouter()


@router.get("/")
def index() -> JSONResponse:
    response = JSONResponse({
        "error": False,
        "message": "Agent API is ready."
    }, status_code=200)

    return response


@router.post("/training/pdf")
def training():
    response = JSONResponse({
        "error": False,
        "status": "live",
        "message": "training by PDF"
    }, status_code=201)

    return response


@router.post("/query/llm")
def askLLM():
    llm_factory = LLMFactory()
    llm_strategy = llm_factory.get_llm_strategy(llm_key="deepseek")
    llm = llm_strategy.get_instance()

    response = JSONResponse({
        "error": False,
        "message": llm
    }, status_code=200)

    return response


@router.post("/llms/rag")
def askRAG():
    llm_factory = LLMFactory()
    llm_strategy = llm_factory.get_llm_strategy(llm_key="deepseek")
    llm = llm_strategy.get_instance()

    response = JSONResponse({
        "error": False,
        "message": llm
    }, status_code=200)

    return response

