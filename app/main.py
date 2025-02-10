import os
import json
import chromadb
import uvicorn

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

load_dotenv(".env")

framework = FastAPI(title="Modular AI Agent")


if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info")
