# src/settings.py

import os
import dotenv

dotenv.load_dotenv()


# ============================
# Environment Variables
# ============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your environment variables."
    )


# ============================
# Model Settings
# ============================

LLM_MODEL = "llama-3.1-8b-instant"

LLM_TEMPERATURE = 0.1


# ============================
# Guardrail Settings
# ============================

BLOCK_OFF_TOPIC = True


# ============================
# Retry Settings
# ============================

MAX_MODEL_RETRIES = 2