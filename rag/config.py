import os
from pathlib import Path

from dotenv import load_dotenv

from .llm import LlmClient
from .store import ChromaStore

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"


def from_env() -> tuple[LlmClient, ChromaStore]:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY가 없습니다. .env에 넣고 다시 실행하세요.")
    llm = LlmClient(
        api_key=api_key,
        embed_model=os.environ.get("EMBED_MODEL", "gemini-embedding-001"),
        chat_model=os.environ.get("CHAT_MODEL", "gemini-3.5-flash-lite"),
    )
    return llm, ChromaStore(str(ROOT / ".chroma"))
