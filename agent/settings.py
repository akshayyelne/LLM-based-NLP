import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
raise ValueError("Missing GROQ_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
