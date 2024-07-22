from fastapi import APIRouter, Request
from services.logger import Logger
from load_llm import LoadLLM

router = APIRouter()
logger = Logger().logger


@router.post("/ai")
def ask_ai(
    request: Request,
    token: str,
    llm: str,
    query: str
):
    logger.info("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    logger.info(f"query: {query}")

    llm = LoadLLM(llm)

    llm_response = llm.invoke(query)
    response = {"answer": llm_response}
    return response
