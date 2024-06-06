import os
import logging
import sys
import tiktoken
import chromadb

from chromadb import Settings
from dotenv import load_dotenv

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.caches import BaseCache
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser, SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough

from models.collection import Collection

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost.com",
    "https://localhost.com",
    "http://localhost",
    "http://localhost:5000",
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s', stream=sys.stdout)

cached_llm = ChatOllama(model="llama3", base_url="http://localhost:11434")
embedding = FastEmbedEmbeddings()
enc = tiktoken.get_encoding("cl100k_base")

def length_function(text: str) -> int:
    return len(enc.encode(text))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_or_create_tenant_for_user(client):
    tenant_id = f"tenant_user:{client}"
    try:
        admin_client.get_tenant(tenant_id)
    except Exception as e:
        admin_client.create_tenant(tenant_id)
    return tenant_id

def get_or_create_db_for_user(category, tenant):
    database = f"db:{category}"
    try:
        admin_client.get_database(database)
    except Exception as e:
        admin_client.create_database(database, tenant)
    return database

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=20,
    length_function=length_function
)

@app.get("/")
def index():
    logging.info("Assistente disponivel")
    response = {"message": "assistente disponivel"}
    return response

@app.post("/ai")
def ask_ai(request: Request):
    logging.info("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    logging.info(f"query: {query}")
    llm_response = cached_llm.invoke(query)
    response = {"answer": llm_response}
    return response

@app.post("/pdf")
async def create_upload_pdf(
    request: Request,
    client: str,
    category: str,
    subject: str,
    file: UploadFile = File(...)
):
    logging.info("Post /pdf called")
    file_name = file.filename
    file_content = await file.read()
    logging.info(f"filename: {file_name}")

    os.makedirs("documents", exist_ok=True)
    tmp_file_name = f"{subject}_{file_name}"
    tmp_file_path = os.path.join("documents", tmp_file_name)

    with open(tmp_file_path, "wb") as f:
        f.write(file_content)

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

@app.post("/ask-pdf")
async def ask_pdf(
    request: Request,
    client: str,
    category: str,
    subject: str,
    query: str,
    prompt: str
):
    logging.info("Post /askPDFPost called")
    logging.info(f"query: {query}")
    logging.info(f"collection: {subject}")

    logging.info(f"Loading Vector Store")
    tenant = get_or_create_tenant_for_user(client)
    database = get_or_create_db_for_user(category, tenant)

    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        tenant=tenant,
        database=database
    )
    vector_store = Chroma(
        client=chroma_client,
        embedding_function=embedding,
        collection_name=subject,
        persist_directory="/chroma/chroma"
    )

    logging.info("Creating chain")

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    logging.info("Retrieved Documents:")
    relevant_documents = retriever.get_relevant_documents(query)

    logging.info("Relevant Documents:")
    references = []
    for doc in relevant_documents:
        reference = {"filename": doc.metadata.get('source'), "content": doc.page_content}
        references.append(reference)

    prompt = PromptTemplate(template=prompt, input_variables=['question'])
    llm_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cached_llm
        | StrOutputParser()
    )
    result = await llm_chain.ainvoke(query)

    logging.info(result)

    response_answer = {
        "answer": result,
        "references": references
    }

    return response_answer

if __name__ == '__main__':
    debug_mode = os.getenv('DEBUG', 'False').lower() in ['true', '1']
    app.run(port=5001, host='0.0.0.0', debug=debug_mode)