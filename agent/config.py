"""Agent configuration."""

import os
from pathlib import Path

# Load .env file manually (no dotenv dependency)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# --- API ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
OPENROUTER_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

# --- Telegram bot ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# --- Memory limits (in tokens, measured via tiktoken cl100k_base) ---
LONG_TERM_MEMORY_MAX_TOKENS = 50_000
WORKING_MEMORY_MAX_TOKENS = 8_000

# --- Retry ---
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds; exponential: base ** attempt
RETRY_JITTER_MAX = 1.0

# --- Knowledge base ---
KNOWLEDGE_BASE_DIR = os.environ.get(
    "KNOWLEDGE_BASE_DIR",
    str(Path(__file__).resolve().parent.parent / "knowledge_base"),
)

# --- Agent ---
MAX_AGENT_TURNS = 50  # safety rail: max tool-call loops per run
