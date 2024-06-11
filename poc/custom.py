from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import StrOutputParser, SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader


import chainlit as cl
import tiktoken
import os

enc = tiktoken.get_encoding("cl100k_base")

def length_function(text: str) -> int:
    return len(enc.encode(text))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOllama(model="llama3", base_url="http://localhost:11434")
vector_db = chroma.Chroma()
vector_db.delete_collection()
retriever = None
history = []


# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# You are a Senior QA Engineer that will write Behat automation tests.
# Use the same structure we use in the context with tags and background.
# Do not write explanations, only a detailed and complete Gherkin code.

# {context}

# Question: {question}
# Helpful Answer:"""
template = """Você é um super assistente, com incríveis habilidade de gestão de projetos
use o contexto para fornecer informaçãoes relevantes e detalhadas sobre as tarefas e sobre
a comunicação entre os membros de equipe da agência CAQO. Não me dê dicas de gestão.
Somente relate o que estã acontecendo de acordo com os elementos que você tem.
Caso eu pergunte isso. Diga para entrar em contato com o administrador.

{context}

Question: {question}
Helpful Answer:"""

@cl.on_chat_start
def main():
    # loader = DirectoryLoader(
    #     './documents/caqo/',
    #     glob='./*.md',
    #     loader_cls=TextLoader
    # )

    loader = TextLoader("./documents/caqo/test2.html")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=650,
        chunk_overlap=100,
        length_function=length_function,
    )

    chunks = text_splitter.split_documents(documents=documents)

    embeddings = FastEmbedEmbeddings()
    vector_docs = vector_db.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    retriever = vector_docs.as_retriever()

    prompt = PromptTemplate(template=template, input_variables=['question'])
    # prompt = SystemMessage(content=template)
    llm_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("llm_chain", llm_chain)
    cl.user_session.set("history", history)

@cl.on_message
async def main(message: cl.Message):

    llm_chain = cl.user_session.get('llm_chain')
    history = cl.user_session.get('history', [])

    history.append(message.content)
    cl.user_session.set('history', history)

    new_input = '\n'.join(history)

    res = await llm_chain.ainvoke(new_input)

    history.append(res)

    await cl.Message(content=str(res)).send()