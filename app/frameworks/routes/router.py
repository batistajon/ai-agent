from fastapi import APIRouter, Request, status
from starlette.responses import JSONResponse

from app.domain.entities.LLMFactory import LLMFactory


router = APIRouter()


@router.get("/")
def index() -> JSONResponse:
    response = JSONResponse({
        "error": False,
        "message": "Agent API is ready."
    }, status_code=200)

    return response


@router.post("/documents")
def documents():
    response = JSONResponse({
        "error": False,
        "status": "live",
        "message": "Documents have been embedded successfully."
    }, status_code=201)

    return response


@router.post("/llms/ask")
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

