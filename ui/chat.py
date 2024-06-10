import streamlit as st
import logging
import random
import time
import json
import requests

st.set_page_config(
    page_title="CAQO - Assistente IA",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.title("🤖 Assistente CAQO")

if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.info("Post /ai called")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Em que posso te ajudar?"):
    logging.info(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_prompt = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            You are a Senior DevOps Engineer with large experience deploying all kinds of applications.
            Please explain to me in detail as if you are teaching a junior developer.

            {context}

            Question: {question}
            Helpful Answer:
        """

        params = {
            "token": "lskdjfhlasdhflaskjdhflaksdjhlfkjasdlkfjahlsdj",
            "llm": "openai",
            "client": "caqo",
            "category": "recursos",
            "subject": "deploy",
            "query": prompt,
            "prompt": response_prompt,
        }

        url = "http://localhost:5000/ask-pdf"
        headers = {
            "Content-Type": "application/json"
        }

        print("Request URL:", url)
        print("Headers:", headers)
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
            time.sleep(0.05)
            response_md.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
