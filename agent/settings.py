# agent/settings.py

import os

# ============================

# Environment Variables

# ============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
raise ValueError(
"GROQ_API_KEY not found. Set it in Streamlit Secrets or environment variables."
)

# ============================

# Model Settings

# ============================

# Recommended stable model (avoid deprecated ones)

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Lower = more deterministic

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))

# ============================

# Guardrail Settings

# ============================

BLOCK_OFF_TOPIC = os.getenv("BLOCK_OFF_TOPIC", "true").lower() == "true"

# ============================

# Retry Settings

# ============================

MAX_MODEL_RETRIES = int(os.getenv("MAX_MODEL_RETRIES", 2))

# ============================

# Debug (optional)

# ============================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
