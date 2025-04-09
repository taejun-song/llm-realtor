import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory
from graph import agentic_rag_pipeline
from utils import get_session_history
# ------------------------------------------------------------------------------
# Streamlit APP
# ------------------------------------------------------------------------------

# Wrap just the `llm` or entire pipeline if you want history across pipeline
chat_with_history: Runnable = RunnableWithMessageHistory(
    agentic_rag_pipeline,
    get_session_history,
    input_messages_key="query",  # This is what the user types
    history_messages_key="messages",  # This needs to be passed if you want history
)


st.set_page_config(page_title="LLM-REALTOR")


st.title("🏠 LLM REALTOR")

user_input = st.chat_input("Ask me about Apartmetns!")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})

    with st.chat_message("user"):
        st.markdown(user_input)
    websites=["apartments.com","realtor.com"]
    graph_input = {
        "query": user_input,
        "websites": websites,
        "messages": st.session_state['messages']  # optional, if you want full chat context
    }
    response = chat_with_history.invoke(graph_input, config={"configurable": {"session_id": "my_unique_user_id"}})['output']



    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state['messages'].append({'role': 'assistant', 'content': response})
