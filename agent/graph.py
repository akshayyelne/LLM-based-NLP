"""
Graph entrypoint for the LangChain Documentation Assistant.

This module exposes the compiled LangGraph agent so other parts
of the application (UI, API, services) can import it without
knowing the internal graph implementation details.
"""

from agent.docs_graph import docs_agent

# Public export of the agent graph
__all__ = ["docs_agent"]
