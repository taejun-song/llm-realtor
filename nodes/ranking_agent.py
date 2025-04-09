from state import PipelineState
from model import llm

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
