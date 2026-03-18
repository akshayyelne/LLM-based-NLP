# agent/settings.py

import os

# ============================

# API KEY

# ============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")

# ============================

# MODEL CONFIG

# ============================

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

# ============================

# FLAGS

# ============================

BLOCK_OFF_TOPIC = True
MAX_MODEL_RETRIES = 2
