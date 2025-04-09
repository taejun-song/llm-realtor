from typing_extensions import TypedDict, List
from langchain_core.documents import Document

class PipelineState(TypedDict):
    query: str
    websites: List[str]
    context: List[Document]
    top_docs: List[Document]
    messages: str
    output: str
