"""
Client LLM thống nhất dùng openai SDK cho cả Ollama (local) và Gemini (cloud).

Trả về kèm thông tin token usage để phục vụ Cost report.
"""
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from engine import config


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


def _ollama_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)


def _gemini_client() -> AsyncOpenAI:
    if not config.has_gemini():
        raise RuntimeError(
            "Thiếu GEMINI_API_KEY. Hãy tạo file .env với dòng: GEMINI_API_KEY=<key của bạn>"
        )
    return AsyncOpenAI(base_url=config.GEMINI_BASE_URL, api_key=config.GEMINI_API_KEY)


def _nvidia_client() -> AsyncOpenAI:
    if not config.has_nvidia():
        raise RuntimeError(
            "Thiếu NVIDIA_API_KEY. Hãy thêm vào .env: NVIDIA_API_KEY=<key của bạn>"
        )
    return AsyncOpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY)


def _openrouter_client() -> AsyncOpenAI:
    if not config.has_openrouter():
        raise RuntimeError(
            "Thiếu OPENROUTER_API_KEY. Hãy thêm vào .env: OPENROUTER_API_KEY=<key của bạn>"
        )
    return AsyncOpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)


async def _chat(client: AsyncOpenAI, model: str, system: str, user: str,
                temperature: float = 0.2, max_retries: int = 4) -> LLMResponse:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            return LLMResponse(
                text=text, model=model, prompt_tokens=pt, completion_tokens=ct,
                cost=config.cost_of(model, pt, ct),
            )
        except Exception as e:  # rate limit / mạng -> exponential backoff
            last_err = e
            if attempt < max_retries - 1:  # không ngủ sau lần thử cuối
                await asyncio.sleep(2.0 * (2 ** attempt))  # 2s, 4s, 8s...
    raise RuntimeError(f"LLM call thất bại sau {max_retries} lần ({model}): {last_err}")


async def ollama_chat(model: str, system: str, user: str, **kw) -> LLMResponse:
    return await _chat(_ollama_client(), model, system, user, **kw)


async def gemini_chat(system: str, user: str, model: Optional[str] = None, **kw) -> LLMResponse:
    return await _chat(_gemini_client(), model or config.GEMINI_MODEL, system, user, **kw)


async def nvidia_chat(system: str, user: str, model: Optional[str] = None, **kw) -> LLMResponse:
    return await _chat(_nvidia_client(), model or config.NVIDIA_MODEL, system, user, **kw)


async def openrouter_chat(system: str, user: str, model: Optional[str] = None, **kw) -> LLMResponse:
    return await _chat(_openrouter_client(), model or config.OPENROUTER_MODEL, system, user, **kw)


def extract_json(text: str) -> dict:
    """Trích object JSON đầu tiên từ output của LLM (có thể lẫn markdown ```json)."""
    text = text.strip()
    # bỏ rào code fence nếu có
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # tìm object {...} cân bằng ngoặc đầu tiên
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Không tìm thấy JSON trong output: {text[:120]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"JSON không cân bằng ngoặc: {text[:120]}")
