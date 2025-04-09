from langchain_core.chat_history import InMemoryChatMessageHistory

def get_session_history(session_id: str):
    return InMemoryChatMessageHistory()
