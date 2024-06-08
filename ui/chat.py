import streamlit as st
import logging
import random
import time
import json
import requests


# Streamed response emulator
def response_generator(bot_response):
    for word in bot_response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Assistente CAQO")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.info("Post /ai called")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    logging.info(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
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
            answer = response_data.get("answer", "")
        except Exception as e:
            response_text = f"Error: {str(e)}"

        response = st.write_stream(response_generator(answer))

    st.session_state.messages.append({"role": "assistant", "content": response})
