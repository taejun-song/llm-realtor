import os
from decouple import Config, RepositoryEnv
from pathlib import Path
from typing_extensions import List, TypedDict
import asyncio

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory
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

class PipelineState(TypedDict):
    query: str
    websites: List[str]
    context: List[Document]
    top_docs: List[Document]
    messages: str
    output: str

# ------------------------------------------------------------------------------
# Agent: Query Refiner
# ------------------------------------------------------------------------------
def query_refiner_node(state: PipelineState) -> PipelineState:
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite vague or incomplete real estate queries into short, specific search queries. "
                "Output ONLY the improved query, no explanation. "
                "Keep it under 20 words. Focus on keywords useful for search engines like location, bedroom count, price, and amenities."
            )
        },
        {"role": "user", "content": f"Original query: {state['query']}"}
    ]
    refined = llm.invoke(messages)
    state["query"] = refined.content.strip().strip('"')
    return state


# ------------------------------------------------------------------------------
# Search + Embed Node
# ------------------------------------------------------------------------------
def duckduckgo_docs(query: str, websites: List[str]) -> List[Document]:
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
    docs = duckduckgo_docs(state['query'], state['websites'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    return state

# ------------------------------------------------------------------------------
# Retriever Node
# ------------------------------------------------------------------------------
def retriever_node(state: PipelineState) -> PipelineState:
    state["context"] = vector_store.similarity_search(state["query"])
    return state

# ------------------------------------------------------------------------------
# Ranking Agent
# ------------------------------------------------------------------------------
def ranking_agent_node(state: PipelineState) -> PipelineState:
    context = state["context"]
    query = state["query"]

    doc_list = "\n\n".join(
        f"[{i}] {doc.page_content}\nURL: {doc.metadata.get('source', '')}"
        for i, doc in enumerate(context)
    )

    messages = [
        {"role": "system", "content": "You are a real estate ranking assistant. Rank the following listings by relevance to the query."},
        {"role": "user", "content": f"Query: {query}\n\n{doc_list}\n\nSelect the top 3 most relevant listings."}
    ]
    response = llm.invoke(messages)

    # Basic: Just take the top 3 documents
    state["top_docs"] = context[:3]
    return state

# ------------------------------------------------------------------------------
# Generator Node
# ------------------------------------------------------------------------------
def generator_node(state: PipelineState) -> PipelineState:
    docs_content = "\n\n".join(
        f"{doc.page_content}\n[Source]({doc.metadata.get('source', '')})"
        for doc in state["top_docs"]
    )
    messages = [
        {
        "role": "system",
        "content": (
            "You are a precise and strict real estate recommendation engine. "
            "You must select **only one** apartment listing that matches the user’s search query **exactly** and use information strictly from the provided crawled context. "
            "You are not allowed to guess, infer, or assume anything. "
            "Only recommend listings that meet all parts of the user's query as explicitly stated in the context. "
            "Do NOT relax the match. For example, if the query asks for '2 bed, 2 bath', you must find a listing that explicitly includes both '2 bed' and '2 bath' in the same description. "
            "Do not recommend listings that say '1 bath' or are missing any required detail. "
            "Do NOT state assumptions like 'likely a typo' or make excuses for mismatched data. "
            "The output must include exactly and only what's in the crawled context: apartment name, address, unit type (must match), and the listing URL (must come from the same listing block). "
            "Never invent or adjust any details. Omit any field that is not present. Do not hallucinate or rationalize missing or conflicting data."
        )
        },
    {
        "role": "user",
        "content": (
            f"{state['query']}\n\n"
            "Below is the crawled context from apartments.com:\n\n"
            f"{docs_content}\n\n"
            "Based **only** on this context, recommend **one** specific apartment listing that matches the query **exactly** (2 bed, 2 bath, located in LA, CA). "
            "Include only the following fields, exactly as found in the same listing: Apartment Name, Address, Apartment Unit Type, Amenities (if any), and URL Link."
            )
        }
    ]
    response = llm.invoke(messages)
    state["output"] = response.content
    return state

# ------------------------------------------------------------------------------
# LangGraph Setup
# ------------------------------------------------------------------------------
graph = StateGraph(PipelineState)

graph.add_node("query_refiner", query_refiner_node)
graph.add_node("duckduckgo_loader", duckduckgo_loader_node)
graph.add_node("retriever", retriever_node)
graph.add_node("ranking_agent", ranking_agent_node)
graph.add_node("generator", generator_node)

graph.set_entry_point("query_refiner")
graph.add_edge("query_refiner", "duckduckgo_loader")
graph.add_edge("duckduckgo_loader", "retriever")
graph.add_edge("retriever", "ranking_agent")
graph.add_edge("ranking_agent", "generator")
graph.set_finish_point("generator")

agentic_rag_pipeline = graph.compile()

def get_session_history(session_id: str):
    return InMemoryChatMessageHistory()

# Wrap just the `llm` or entire pipeline if you want history across pipeline
chat_with_history: Runnable = RunnableWithMessageHistory(
    agentic_rag_pipeline,
    get_session_history,
    input_messages_key="query",  # This is what the user types
    history_messages_key="messages",  # This needs to be passed if you want history
)


st.set_page_config(page_title="LLM-REALTOR")


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
    graph_input = {
        "query": user_input,
        "websites": websites,
        "messages": st.session_state['messages']  # optional, if you want full chat context
    }
    response = chat_with_history.invoke(graph_input, config={"configurable": {"session_id": "my_unique_user_id"}})['output']



    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state['messages'].append({'role': 'assistant', 'content': response})
