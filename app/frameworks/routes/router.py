from fastapi import APIRouter, Request, status
from starlette.responses import JSONResponse


router = APIRouter()


@router.get("/")
def index() -> JSONResponse:
    response = JSONResponse({
        "error": False,
        "status": "live",
        "message": "Agent API is ready."
    }, status_code=200)

    return response


@router.get("/documents")
def documents():
    response = JSONResponse({
        "error": False,
        "status": "live",
        "message": "Documents have been embedded successfully."
    }, status_code=201)

    return response

