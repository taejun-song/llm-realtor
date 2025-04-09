from state import PipelineState
from model import vector_store

# ------------------------------------------------------------------------------
# Retriever Node
# ------------------------------------------------------------------------------
def retriever_node(state: PipelineState) -> PipelineState:
    state["context"] = vector_store.similarity_search(state["query"])
    return state
