import os
from decouple import Config, RepositoryEnv
from pathlib import Path
from typing_extensions import List, TypedDict

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
# ------------------------------------------------------------------------------
# Environment & Global Setup
# ------------------------------------------------------------------------------

root_dir = Path().resolve()

config = Config(RepositoryEnv(root_dir / '.env'))  # Explicitly load .env

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = config('LANGSMITH_API_KEY')

llm = ChatOllama(model="llama3.2",temperature=0)

embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)

# State Typing
class PipelineState(TypedDict):
    query: str
    websites: List[str]
    context: List[Document]
    answer: str

# ------------------------------------------------------------------------------
# DuckDuckGO Loader
# ------------------------------------------------------------------------------

def duckduckgo_docs(query: str, websites: List[str]) -> List[Document]:
    """
    Search DuckDuckGo for a query restricted to specific websites.

    Args:
        query (str): The search query string.
        websites (List[str]): A list of website domains to search within.

    Returns:
        List[Document]: A list of LangChain Document objects created from search result snippets.
    """
    search = DuckDuckGoSearchResults(output_format="list")
    results = [
        item
        for website in websites
        for item in search.invoke(f"{query} site:{website}")
    ]
    return [
        Document(
            page_content=entry["snippet"],
            metadata={"source": entry["link"]}
        )
        for entry in results
    ]


def duckduckgo_loader_node(state: PipelineState) -> PipelineState:
    """
    Load DuckDuckGo search results into a vector store after chunking text.

    Args:
        state (PipelineState): A pipeline state dictionary containing:
            - 'query': str, the search query.
            - 'websites': List[str], the domains to search.

    Returns:
        PipelineState: The updated pipeline state.
    """
    docs = duckduckgo_docs(state['query'], state['websites'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)

    return state

def retriever_node(state: PipelineState):
    # Placeholder for actual retrieval logic
    state["context"] = vector_store.similarity_search(state["query"])
    return state

def generator_node(state: PipelineState):
    # Placeholder for actual generation logic
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly confident, precise real estate recommendation engine. "
                "Based solely on the provided crawled context, you must select one specific condo listing that best meets the search criteria. "
                "Even if some details are missing, combine the available information to provide a clear, detailed recommendation. "
                "Do NOT state that you cannot recommend or express uncertainty. "
                "Always output a recommendation that includes the exact address, unit type, and amenities as they appear in the context. "
                "If a particular detail is not mentioned, simply omit it—do not add or hallucinate any information."
            )
        },
        {
            "role": "user",
            "content": (
                "Below is the crawled context from apartments.com:\n\n"
                f"{docs_content}\n\n"
                "Based solely on this data, please provide a detailed recommendation for one specific condo listing that meets the search criteria. "
                "Include all available details (e.g. address, unit type, and amenities) exactly as found in the context."
            )
        }
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state



# Graph definition
graph = StateGraph(PipelineState)
graph.add_node("duckduckgo_loader", duckduckgo_loader_node)
graph.add_node("retriever", retriever_node)
graph.add_node("generator", generator_node)

graph.set_entry_point("duckduckgo_loader")
graph.add_edge("duckduckgo_loader", "retriever")
graph.add_edge("retriever", "generator")
graph.set_finish_point("generator")

# Compile Graph
agentic_rag_pipeline = graph.compile()


st.set_page_config(page_title="LLM-REALTOR")

if 'conversation' not in st.session_state:
    memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory
    )

st.title("🏠 LLM REALTOR")

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
    websites=["apartments.com","realtor.com"]
    graph_input = {"query": user_input, "websites":websites}
    response = agentic_rag_pipeline.invoke(graph_input)['answer']

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state['messages'].append({'role': 'assistant', 'content': response})
