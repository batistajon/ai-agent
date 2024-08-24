import os
import json

from fastapi import HTTPException
import chromadb
import uvicorn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from app.helpers.common import length_function, format_docs
from app.services.logger import Logger

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

from fastapi import FastAPI
from fastapi.responses import FileResponse

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
# openai_llm = ChatOpenAI(model="gpt-4o")
openai_llm = ChatOpenAI(model="text-ada-002")


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
    logger.info("Assistente disponivel")
    response = {"message": "assistente disponivel"}
    return response


@app.post("/ai")
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
    chunk_size: int = 1024,
    chunk_overlap: int = 450,
    file: UploadFile = File(...)
):
    with open("token.json", 'r') as token_json:
        clients = json.load(token_json)
        if clients.get("client") != client:
            raise Exception("Cliente inválido!")
        if clients.get("token") != token:
            raise Exception("Token inválido!")

    logger.info("Post /pdf called")
    file_name = file.filename
    file_content = await file.read()
    logger.info(f"filename: {file_name}")

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
    logger.info(f"docs: {len(docs)}")

    chunks = text_splitter.split_documents(docs)
    logger.info(f"chunks len: {len(chunks)}")

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
    logger.info("Vector store created successfully.")

    os.remove(tmp_file_path)

    response = {
        "status": "Successfully Uploaded",
        "arquivo": file_name,
        "cliente": client,
        "categoria": category,
        "assunto": subject,
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
    logger.info("Post /askPDFPost called")
    with open("token.json", 'r') as token_json:
        clients = json.load(token_json)
        if clients.get("client") != request.client:
            raise Exception("Cliente inválido!")
        if clients.get("token") != request.token:
            raise Exception("Token inválido!")

    logger.info(f"query: {request.query}")
    logger.info(f"collection: {request.subject}")

    logger.info(f"Loading Vector Store")
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
        search_kwargs={"k": 20, "score_threshold": 0.05},  # Ajustado para 0.05
    )


    logger.info(f"Documentos retornados: {retriever}")
    relevant_documents = retriever.get_relevant_documents(request.query)

    logger.info(f"Documentos relevantes: {relevant_documents}")
    references = []
    for doc in relevant_documents:
        reference = {"filename": doc.metadata.get(
            'source'), "content": doc.page_content}
        references.append(reference)

    prompt = PromptTemplate(template=request.prompt,
                            input_variables=['question'])
    logger.info("Creating chain")
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

    logger.info(result)

    response_answer = {
        "answer": result,
        "references": references
    }

    return response_answer

######### DAQUI PRA BAIXO SÃO SUGESTÕES!!
# Função em construção, visando listar, os tenants criados
@app.get("/list-tenants", summary="Lista todos os tenants e suas collections", description="Retorna uma lista de todos os tenants disponíveis no sistema e suas respectivas collections.")
def list_tenants():
    logger.info("Listando todos os tenants e suas collections")
    try:
        # Lista todos os tenants disponíveis
        tenants = admin_client.list_tenants()
        tenant_collections = {}
        for tenant in tenants:
            # Obter o nome do tenant atual
            tenant_name = tenant['name']
            # Obter todas as databases para o tenant atual
            databases = admin_client.list_databases(tenant_name)
            collections = {}
            for db in databases:
                # Obter o nome da database
                db_name = db['name']
                # Listar todas as collections para a database
                db_collections = admin_client.list_collections(tenant_name, db_name)
                collections[db_name] = db_collections
            tenant_collections[tenant_name] = collections
        return {"tenants": tenants, "tenant_collections": tenant_collections}
    except Exception as e:
        logger.error(f"Erro ao listar tenants e suas collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-pdfs")
async def list_pdfs(tenant_name: str):
    try:
        # Conectar ao cliente ChromaDB
        chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            tenant="tenant_caqo",
            database="db_plataforma_networking"  # Ajuste conforme a necessidade
        )
        
        # Obter todas as coleções da database especificada
        collections = admin_client.list_collections(tenant_name, "db_plataforma_networking")
        
        # Filtrar documentos que são PDFs
        pdf_documents = []
        for collection in collections:
            documents = chroma_client.list_documents(collection['name'])
            for doc in documents:
                if doc['name'].endswith('.pdf'):  # Verifica se o nome do documento termina com '.pdf'
                    pdf_documents.append(doc)
        
        if not pdf_documents:
            return {"message": "No PDFs found in the specified tenant."}
        
        return {"pdfs": pdf_documents}
    
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

