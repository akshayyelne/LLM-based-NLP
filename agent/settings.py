import os
import streamlit as st

# Try Streamlit secrets first, fallback to env
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
