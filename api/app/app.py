import os
import logging
import sys
import tiktoken
import chromadb

from chromadb.config import Settings
from dotenv import load_dotenv

from flask import Flask, request
from flask_cors import CORS

from langchain_core.caches import BaseCache
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app)

# chroma_client = chromadb.HttpClient(
#     host="localhost",
#     port=8000
# )

logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s', stream=sys.stdout)

cached_llm = Ollama(model="llama3", base_url="http://localhost:11434")
embedding = FastEmbedEmbeddings()

enc = tiktoken.get_encoding("cl100k_base")
def length_function(text: str) -> int:
    return len(enc.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1024,
    chunk_overlap=80,
    length_function=length_function
)

raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route("/", methods=["GET"])
def index():
    logging.info("Assistente disponivel")
    response = {"message": "assistente disponivel 3"}
    return response

@app.route("/ai", methods=["POST"])
def aiPost():
    logging.info("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    logging.info(f"query: {query}")
    llm_response = cached_llm.invoke(query)
    response = {"answer": llm_response}
    return response

@app.route("/pdf", methods=["POST"])
def pdfPost():
    logging.info("Post /pdfPost called")
    collection = request.form["collection"]

    file = request.files["file"]
    file_name = file.filename
    save_file = "documents/" + file_name
    file.save(save_file)

    logging.info(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    logging.info(f"docs: {len(docs)}")

    chunks = text_splitter.split_documents(documents=docs)
    logging.info(f"chunks len: {len(chunks)}")

    # vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, client=chroma_client, collection_name=collection, persist_directory="/chroma/chroma")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, collection_name=collection, persist_directory="db")
    logging.info("Vector store created successfully.")

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks_len": len(chunks)
    }

    return response

@app.route("/ask-pdf", methods=["POST"])
def askPDFPost():
    logging.info("Post /askPDFPost called")
    json_content = request.json
    query = json_content.get("query")
    collection = json_content.get("collection")
    logging.info(f"query: {query}")
    logging.info(f"collection: {collection}")

    logging.info(f"Loading Vector Store")
    # vector_store = Chroma(client=chroma_client, embedding_function=embedding, collection_name=collection, persist_directory="/chroma/chroma")
    vector_store = Chroma(embedding_function=embedding, collection_name=collection, persist_directory="db")

    logging.info("Creating chain")

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    logging.info(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


if __name__ == '__main__':
    debug_mode = os.getenv('DEBUG', 'False').lower() in ['true', '1']
    app.run(port=5001, host='0.0.0.0', debug=debug_mode)