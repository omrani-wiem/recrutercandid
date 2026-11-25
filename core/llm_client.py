"""Client LLM unique (API compatible OpenAI, par défaut Ollama en local)."""
import os
from dotenv import load_dotenv

load_dotenv() 

try:
    from langfuse.openai import OpenAI  
except ImportError:
    from openai import OpenAI            
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:7b")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
    max_retries=1,
)