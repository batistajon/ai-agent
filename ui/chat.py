import streamlit as st
import pandas as pd
import logging
import random
import time
import json
import requests


st.set_page_config(
    page_title="CAQO - Assistente IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Assistente da CAQO!"
    }
)

st.title("🤖 Assistente CAQO")

st.sidebar.header("Customize seu assistente")

def show_alert():
    st.warning('Este é um aviso de teste!')
    st.error('Este é um erro de teste!')
    st.success('Esta é uma confirmação de teste!')

import streamlit as st

css='''
<style>
[data-testid="stFileUploaderDropzone"] div div::before {color:green; content:"Clique aqui ou Arraste os arquivos"}
[data-testid="stFileUploaderDropzone"] div div span{display:none;}
[data-testid="stFileUploaderDropzone"] div div::after {color:green; font-size: .8em; content:"Até 10MB por arquivo"}
[data-testid="stFileUploaderDropzone"] div div small{display:none;}
section[data-testid="stFileUploaderDropzone"] > button {
display: none;
}

</style>
'''

st.markdown(css, unsafe_allow_html=True)

with st.sidebar.expander("Parâmetros"):
    category = st.text_input("Categoria", help="Organize seu assistente por Categorias")
    subject = st.text_input("Assunto", value="assunto_teste", help="Escolha um assunto para deixar o assistente mais eficiente.")
    llm_choice = st.selectbox("Escolha a LLM", ["Premium", "Free"], help="")

with st.sidebar.expander("Importar"):
    uploaded_files = st.file_uploader("Suba um PDF para fazer perguntas", accept_multiple_files=True, help="A IA irá te responder sobre o conteúdo dos arquivos")
    if uploaded_files:
        st.write("Arquivos carregados!")

with st.sidebar.expander("Não mexer"):
    token = st.text_input("Token (não mexer)", help="Token de autenticação para a API.", value="lskdjfhlasdhflaskjdhflaksdjhlfkjasdlkfjahlsdj")
    client = st.text_input("Cliente", value="caqo", help="Identificador do cliente usando a API.")
    if st.button('Mostrar alerta'):
        show_alert()


def embed_files():
    for uploaded_file in uploaded_files:
        # Preparando o arquivo para ser enviado
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        # Preparando parâmetros de consulta
        params = {
            "token": token,
            "client": client,
            "category": category,
            "subject": subject,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        # URL para a requisição incluindo parâmetros de consulta
        url = "http://localhost:5000/pdf"
        
        # Realizando o pedido POST com parâmetros de consulta
        response = requests.post(url, params=params, files=files)
        
        # Verificando a resposta do servidor
        if response.status_code == 200:
            st.success("Arquivo enviado e processado com sucesso!")
            st.json(response.json())  # Exibindo a resposta JSON
        else:
            st.error("Falha ao enviar arquivo: " + response.text)
            print(response.status_code, response.text)  # Para debug


if uploaded_files:
    token = st.sidebar.text_input("Token (não mexer)", help="Token de autenticação para a API.", value="lskdjfhlasdhflaskjdhflaksdjhlfkjasdlkfjahlsdj")
    client = st.sidebar.text_input("Cliente", value="caqo", help="Identificador do cliente usando a API.")
    category = st.sidebar.text_input("Categoria", help="Organize seu assistente por Categorias")
    subject = st.sidebar.text_input("Assunto", value="assunto_teste", help="Escolha um assunto para deixar o assistente mais eficiente.")
    chunk_size = st.sidebar.number_input("Defina o tamanho do chunk", min_value=100, max_value=2048, value=800, step=50, help="Define o tamanho de cada pedaço do documento para análise.")
    chunk_overlap = st.sidebar.number_input("Defina a sobreposição do chunk", min_value=0, max_value=500, value=150, step=10, help="Define a sobreposição entre os chunks para evitar a perda de contexto.")
    llm_choice = st.sidebar.selectbox("Escolha a LLM", {"Premium", "Free"}, help="")
    if llm_choice == "Premium":
        llm = "openai"
    else:
        llm = "ollama"


    doc_ext_options = ["PDF", "Word", "CSV", "URL"]
    doc_ext_mapping = {
        "PDF": "pdf",
        "Word": "text",
        "CSV": "csv",
        "URL": "url"
    }

    doc_size_options = [
        "Menos de 10 Paginas",
        "De 11 a 50 paginas",
        "De 51 a 100 paginas",
        "Mais de 100"
    ]
    doc_size_mapping = {
        "Menos de 10 Paginas": (150, 20),
        "De 11 a 50 paginas": (300, 40),
        "De 51 a 100 paginas": (450, 80),
        "Mais de 100": (1024, 200)
    }
    selected_doc_ext = st.sidebar.selectbox("Escolha o Tipo do Arquivo", doc_ext_options)
    selected_doc_size = st.sidebar.selectbox("Qual o Tamanho do Arquivo em Paginas", doc_size_options)

    doc_ext_value = doc_ext_mapping[selected_doc_ext]
    doc_size_value = doc_size_mapping[selected_doc_size]
    st.sidebar.button(label="Treinar IA", on_click=embed_files)


if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.info("Post /ai called")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Como posso ajudar?"):
    logging.info(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        response_prompt = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            You are a Senior DevOps Engineer with large experience deploying all kinds of applications.
            Please explain to me in detail as if you are teaching a junior developer. Important that your answer be in portuguese

            {context}

            Question: {question}
            Helpful Answer:
        """

        params = {
            "token": token,
            "llm": llm,
            "client": "tenant_caqo",
            "category": category,
            "subject": subject,
            "query": prompt,
            "prompt": response_prompt,
        }

        url = "http://localhost:5000/ask-pdf"
        headers = {
            "Content-Type": "application/json"
        }

        print("Parameters:", json.dumps(params, indent=2))
        json_data = json.dumps(params)
        answer=""

        try:
            res = requests.post(url=url, data=json_data, headers=headers)
            response_text = res.text
            response_data = json.loads(response_text)
            answer = response_data.get("answer")
        except Exception as e:
            response_text = f"Error: {str(e)}"

        print(answer)
        response = ""
        response_md = st.empty()
        for chunk in answer:
            response += chunk
            time.sleep(0.01)
            response_md.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
