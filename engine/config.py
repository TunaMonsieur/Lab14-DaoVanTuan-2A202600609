"""
Cấu hình tập trung cho Evaluation Factory.

- Nạp biến môi trường từ .env (nếu có).
- Định nghĩa endpoint/model cho Gemini (cloud) và Ollama (local).
- Bảng giá token để tính Cost report.
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def setup_utf8():
    """Bảo đảm stdout/stderr in được tiếng Việt trên Windows console (cp1252)."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


# --- Gemini (cloud, qua endpoint tương thích OpenAI) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# --- OpenRouter (cloud, qua endpoint tương thích OpenAI) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

# --- NVIDIA NIM (cloud, tùy chọn - free-tier rate-limit thấp) ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")

# --- Ollama (local, qua endpoint tương thích OpenAI) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = "ollama"  # placeholder, Ollama không kiểm tra
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "kamekichi128/qwen3-4b-instruct-2507:latest")
AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.2:1b")

# --- Bảng giá (USD / 1 triệu token) ---
# Ollama chạy local => miễn phí. Gemini Flash giá tham khảo theo Google AI pricing.
PRICING = {
    "gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "deepseek-ai/deepseek-v4-flash": {"input": 0.07, "output": 0.27},
    "deepseek-ai/deepseek-v4-pro":   {"input": 0.27, "output": 1.10},
    "openai/gpt-oss-120b:free": {"input": 0.0, "output": 0.0},  # OpenRouter free tier
    "openai/gpt-oss-120b":      {"input": 0.09, "output": 0.45},  # giá tham khảo bản trả phí
    "llama3.2:1b":      {"input": 0.0,  "output": 0.0},
    "kamekichi128/qwen3-4b-instruct-2507:latest": {"input": 0.0, "output": 0.0},
    "qwen3:4b":         {"input": 0.0,  "output": 0.0},
    "qwen2.5-coder:7b": {"input": 0.0,  "output": 0.0},
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


def has_gemini() -> bool:
    return bool(GEMINI_API_KEY)


def has_nvidia() -> bool:
    return bool(NVIDIA_API_KEY)


def has_openrouter() -> bool:
    return bool(OPENROUTER_API_KEY)
