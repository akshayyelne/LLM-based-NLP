# LangGraph Documentation Assistant

import asyncio
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langchain.tools import tool

from langgraph.graph import StateGraph, END

from agent.config import llm
from middleware.guardrails_middleware import GuardrailsMiddleware
from prompts.docs_agent_prompt import docs_agent_prompt

from tools.docs_tools import SearchDocsByLangChain
from tools.support_tools import search_support_articles, get_article_content
from tools.link_check_tools import (
    _check_urls_async,
    _format_results,
    DEFAULT_TIMEOUT,
)


# -------------------------
# Tool Wrappers
# -------------------------

@tool
def search_support_articles_tool(query: str) -> str:
    """Search LangChain support articles for the given query."""
    return search_support_articles(query)


@tool
def get_article_content_tool(article_id: str) -> str:
    """Get the full content of a support article by its ID."""
    return get_article_content(article_id)


@tool
def check_links_tool(url: str) -> str:
    """Check if a URL is valid and accessible. Use this whenever the user asks to validate a link.

    Args:
        url: The URL to validate (e.g. https://python.langchain.com).

    Returns:
        Whether the link is valid, its status code, and any redirect info.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_check_urls_async([url], DEFAULT_TIMEOUT))
        finally:
            loop.close()
        return _format_results(results)
    except Exception as e:
        return f"Error checking URL '{url}': {str(e)}"


# -------------------------
# Tools
# -------------------------

tools = [
    SearchDocsByLangChain,
    search_support_articles_tool,
    get_article_content_tool,
    check_links_tool,
]

tool_map = {t.name: t for t in tools if hasattr(t, "name")}

model = llm.bind_tools(tools)


# -------------------------
# State — messages uses add_messages so they accumulate across nodes
# -------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    blocked: bool
    tool_used: bool


# -------------------------
# Guardrails
# -------------------------

guardrails = GuardrailsMiddleware(model=llm, block_off_topic=True)


async def guardrails_node(state: AgentState, config):
    result = await guardrails.abefore_agent(state)

    if result:
        return {
            "messages": result.get("messages", []),
            "blocked": True,
        }

    return {"blocked": False}


def guardrails_router(state: AgentState):
    return END if state.get("blocked") else "agent"


# -------------------------
# Agent
# -------------------------

safe_prompt = docs_agent_prompt.replace("{", "{{").replace("}", "}}")

prompt = ChatPromptTemplate.from_messages([
    ("system", safe_prompt),
    MessagesPlaceholder("messages"),
])
agent_chain = prompt | model


async def agent_node(state: AgentState, config):
    try:
        response = await agent_chain.ainvoke({
            "messages": state["messages"]
        })

        has_content = bool(getattr(response, "content", None))
        has_tool_calls = bool(getattr(response, "tool_calls", None))

        if not response or (not has_content and not has_tool_calls):
            return {
                "messages": [
                    ToolMessage(
                        tool_call_id="fallback",
                        content="I was unable to generate a response. Please try rephrasing your question."
                    )
                ]
            }

        return {"messages": [response]}

    except Exception as e:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id="error",
                    content=f"Error: {str(e)}"
                )
            ]
        }


# -------------------------
# Tool Execution
# -------------------------

async def tools_node(state: AgentState):
    last = state["messages"][-1]

    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return {}

    tool_messages = []

    for tool_call in tool_calls:
        t = tool_map.get(tool_call["name"])

        if not t:
            result = f"Tool '{tool_call['name']}' not found"
        else:
            try:
                result = await asyncio.to_thread(
                    t.invoke,
                    tool_call.get("args", {})
                )
            except Exception as e:
                result = f"Tool error: {str(e)}"

        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(result),
            )
        )

    return {
        "messages": tool_messages,
        "tool_used": True
    }


# -------------------------
# Router
# -------------------------

def tool_router(state: AgentState):
    last = state["messages"][-1]

    if state.get("tool_used"):
        return END

    if getattr(last, "tool_calls", None):
        return "tools"

    return END


# -------------------------
# Graph
# -------------------------

workflow = StateGraph(AgentState)

workflow.add_node("guardrails", guardrails_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.set_entry_point("guardrails")

workflow.add_conditional_edges(
    "guardrails",
    guardrails_router,
    {"agent": "agent", END: END},
)

workflow.add_conditional_edges(
    "agent",
    tool_router,
    {"tools": "tools", END: END},
)

workflow.add_edge("tools", "agent")

docs_agent = workflow.compile()
