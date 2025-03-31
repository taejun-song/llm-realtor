import os
from decouple import Config, RepositoryEnv
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ------------------------------------------------------------------------------
# Environment & Global Setup
# ------------------------------------------------------------------------------

root_dir = Path().resolve()
print(root_dir / '.env')

config = Config(RepositoryEnv(root_dir / '.env'))  # Explicitly load .env

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = config('LANGSMITH_API_KEY')


st.set_page_config(page_title="LLM-REALTOR")

if 'conversation' not in st.session_state:
    memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(
        llm=ChatOllama(model="llama3.2",temperature=0),
        memory=memory
    )

st.title("💬 LLM REALTOR")

user_input = st.chat_input("Ask me anything!")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = st.session_state.conversation.predict(input=user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state['messages'].append({'role': 'assistant', 'content': response})
