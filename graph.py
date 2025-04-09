from langgraph.graph import StateGraph
from state import PipelineState
from nodes.query_refiner import query_refiner_node
from nodes.retriever import retriever_node
from nodes.ranking_agent import ranking_agent_node
from nodes.generator import generator_node
from nodes.duckduckgo_loader import duckduckgo_loader_node

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
