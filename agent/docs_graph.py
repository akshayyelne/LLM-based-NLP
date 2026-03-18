# agent/docs_graph.py

import asyncio
from typing import TypedDict, List

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

from langgraph.graph import StateGraph, END

from agent.config import llm
from middleware.guardrails_middleware import GuardrailsMiddleware
from prompts.docs_agent_prompt import docs_agent_prompt

from tools.docs_tools import SearchDocsByLangChain
from tools.support_tools import search_support_articles, get_article_content
from tools.link_check_tools import check_links


# -------------------------
# Tool Wrappers
# -------------------------

@tool
def search_support_articles_tool(query: str):
    """Search support articles."""
    return search_support_articles(query)


@tool
def get_article_content_tool(article_id: str):
    """Get article content."""
    return get_article_content(article_id)


@tool
def check_links_tool(url: str):
    """Validate URL."""
    return check_links(url)


# -------------------------
# Tools (IMPORTANT FIX)
# -------------------------

tools = [
    SearchDocsByLangChain,   # ✅ NO ()
    search_support_articles_tool,
    get_article_content_tool,
    check_links_tool,
]

tool_map = {t.name: t for t in tools if hasattr(t, "name")}

model = llm.bind_tools(tools)


# -------------------------
# State
# -------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]
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

prompt = ChatPromptTemplate.from_messages([
    ("system", docs_agent_prompt),  # ✅ NO escaping
    MessagesPlaceholder("messages"),
])

agent_chain = prompt | model


async def agent_node(state: AgentState, config):
    try:
        response = await agent_chain.ainvoke({
            "messages": state["messages"]
        })

        # ✅ fallback if empty
        if not response or not getattr(response, "content", None):
            fallback = await llm.ainvoke(state["messages"])
            return {"messages": [fallback]}

        return {"messages": [response]}

    except Exception:
        fallback = await llm.ainvoke(state["messages"])
        return {"messages": [fallback]}


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
        tool = tool_map.get(tool_call["name"])

        if not tool:
            result = "Tool not found"
        else:
            try:
                result = await asyncio.to_thread(
                    tool.invoke,
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
        "tool_used": True  # prevents loop
    }


# -------------------------
# Router
# -------------------------

def tool_router(state: AgentState):
    last = state["messages"][-1]

    # stop loop after one tool call
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
