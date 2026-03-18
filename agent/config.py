# agent/config.py
import logging

from langchain_groq import ChatGroq

from agent.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

# ============================
# Logger
# ============================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("docs-agent")


# ============================
# LLM Initialization
# ============================

llm = ChatGroq(
    model_name=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    groq_api_key=GROQ_API_KEY,
)

logger.info(f"Groq LLM initialized: {LLM_MODEL}")


__all__ = [
    "llm",
    "logger",
]
