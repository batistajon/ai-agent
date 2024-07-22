import os
import json
import chromadb
import uvicorn

from helpers.common import length_function, format_docs
from services.logger import Logger

from chromadb import Settings
from dotenv import load_dotenv

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser, SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough

load_dotenv(".env")

app = FastAPI(title="Assistente CAQO")

origins = [
    "http://localhost.com",
    "https://localhost.com",
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:8501",
    "http://localhost:8501",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

admin_client = chromadb.AdminClient(Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    chroma_server_host="localhost",
    chroma_server_http_port="8000",
))

logger = Logger().logger

# Redirect to routes ----------
# Everything underneath this lines should be abstacted

ollama_llm = ChatOllama(model="llama3", base_url="http://localhost:11434")
openai_llm = ChatOpenAI(model="gpt-4-1106-preview")

embedding = FastEmbedEmbeddings()


def get_or_create_tenant_for_user(client):
    tenant_id = f"tenant_{client}"
    try:
        admin_client.get_tenant(tenant_id)
    except Exception as e:
        admin_client.create_tenant(tenant_id)
    return tenant_id


def get_or_create_db_for_user(category, tenant):
    database = f"db_{category}"
    try:
        admin_client.get_database(database)
    except Exception as e:
        admin_client.create_database(database, tenant)
    return database


@app.get("/")
def index():
    logging.info("Assistente disponivel")
    response = {"message": "assistente disponivel"}
    return response


@app.post("/ai")
def ask_ai(
    request: Request,
    token: str,
    llm: str,
    query: str
):
    logging.info("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    logging.info(f"query: {query}")

    cached_llm = ""
    if request.llm == "openai":
        cached_llm = openai_llm
    else:
        cached_llm = ollama_llm

    llm_response = cached_llm.invoke(query)
    response = {"answer": llm_response}
    return response


@app.post("/pdf")
async def create_upload_pdf(
    request: Request,
    token: str,
    client: str,
    category: str,
    subject: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    file: UploadFile = File(...)
):
    with open("token.json", 'r') as token_json:
        clients = json.load(token_json)
        if clients.get("client") != client:
            raise Exception("Invalid client")
        if clients.get("token") != token:
            raise Exception("Invalid token")

    logging.info("Post /pdf called")
    file_name = file.filename
    file_content = await file.read()
    logging.info(f"filename: {file_name}")

    os.makedirs("documents", exist_ok=True)
    tmp_file_name = f"{subject}_{file_name}"
    tmp_file_path = os.path.join("documents", tmp_file_name)

    with open(tmp_file_path, "wb") as f:
        f.write(file_content)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function
    )
    loader = PDFPlumberLoader(tmp_file_path)
    docs = loader.load_and_split()
    logging.info(f"docs: {len(docs)}")

    chunks = text_splitter.split_documents(docs)
    logging.info(f"chunks len: {len(chunks)}")

    tenant = get_or_create_tenant_for_user(client)
    database = get_or_create_db_for_user(category, tenant)

    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        tenant=tenant,
        database=database
    )
    chroma_client.get_or_create_collection(subject)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        client=chroma_client,
        collection_name=subject,
        persist_directory="/chroma/chroma"
    )
    logging.info("Vector store created successfully.")

    os.remove(tmp_file_path)

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks_len": len(chunks)
    }

    return response


class AskPDFRequest(BaseModel):
    token: str
    llm: str = "ollama"
    client: str
    category: str
    subject: str
    query: str
    prompt: str


@app.post("/ask-pdf")
async def ask_pdf(
    request: AskPDFRequest
):
    logging.info("Post /askPDFPost called")
    with open("token.json", 'r') as token_json:
        clients = json.load(token_json)
        if clients.get("client") != request.client:
            raise Exception("Invalid client")
        if clients.get("token") != request.token:
            raise Exception("Invalid token")

    logging.info(f"query: {request.query}")
    logging.info(f"collection: {request.subject}")

    logging.info(f"Loading Vector Store")
    tenant = admin_client.get_tenant(f"tenant_{request.client}")
    database = admin_client.get_database(
        f"db_{request.category}", tenant['name'])

    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        tenant=tenant['name'],
        database=database['name']
    )
    vector_store = Chroma(
        client=chroma_client,
        embedding_function=embedding,
        collection_name=request.subject,
        persist_directory="/chroma/chroma"
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    logging.info(f"Retrieved Documents: {retriever}")
    relevant_documents = retriever.get_relevant_documents(request.query)

    logging.info(f"Relevant Documents: {relevant_documents}")
    references = []
    for doc in relevant_documents:
        reference = {"filename": doc.metadata.get(
            'source'), "content": doc.page_content}
        references.append(reference)

    prompt = PromptTemplate(template=request.prompt,
                            input_variables=['question'])
    logging.info("Creating chain")
    cached_llm = ""
    if request.llm == "openai":
        cached_llm = openai_llm
    else:
        cached_llm = ollama_llm

    llm_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cached_llm
        | StrOutputParser()
    )
    result = await llm_chain.ainvoke(request.query)

    logging.info(result)

    response_answer = {
        "answer": result,
        "references": references
    }

    return response_answer

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
