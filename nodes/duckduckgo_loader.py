from typing_extensions import List, TypedDict

from state import PipelineState
from model import llm
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model import vector_store

# ------------------------------------------------------------------------------
# Search + Embed Node
# ------------------------------------------------------------------------------

def duckduckgo_docs(query: str, websites: list[str]) -> list[Document]:
    search = DuckDuckGoSearchResults(output_format="list")
    results = [item for site in websites for item in search.invoke(f"{query} site:{site}")]
    return [Document(page_content=entry["snippet"], metadata={"source": entry["link"]}) for entry in results]


def duckduckgo_loader_node(state: PipelineState) -> PipelineState:
    docs = duckduckgo_docs(state['query'], state['websites'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    return state
