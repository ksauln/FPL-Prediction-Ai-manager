"""HTTP client for a local Ollama-compatible chat endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional

import requests


class LocalLLMError(RuntimeError):
    """Raised when the local LLM request fails."""


@dataclass
class OllamaSettings:
    """Connection settings for a local Ollama-compatible endpoint."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout_seconds: int = 120
    temperature: float = 0.2


def chat_with_local_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: OllamaSettings,
) -> str:
    """Send a chat request to a local Ollama-compatible server."""

    url = settings.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": settings.model,
        "stream": False,
        "options": {"temperature": settings.temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        response = requests.post(url, json=payload, timeout=settings.timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LocalLLMError(
            f"Failed to reach the local LLM endpoint at {url}: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise LocalLLMError("Local LLM returned a non-JSON response.") from exc

    message: Optional[dict] = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not content:
        raise LocalLLMError("Local LLM response did not include message content.")
    return str(content).strip()


def stream_chat_with_local_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: OllamaSettings,
) -> Iterator[str]:
    """Stream a chat response from a local Ollama-compatible server."""

    url = settings.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": settings.model,
        "stream": True,
        "options": {"temperature": settings.temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        with requests.post(
            url,
            json=payload,
            timeout=settings.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            yielded_any = False
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise LocalLLMError("Local LLM returned malformed stream data.") from exc

                if data.get("error"):
                    raise LocalLLMError(str(data["error"]))

                message = data.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        yielded_any = True
                        yield str(content)

                if data.get("done"):
                    break

            if not yielded_any:
                raise LocalLLMError("Local LLM stream did not include message content.")
    except requests.RequestException as exc:
        raise LocalLLMError(
            f"Failed to reach the local LLM endpoint at {url}: {exc}"
        ) from exc
