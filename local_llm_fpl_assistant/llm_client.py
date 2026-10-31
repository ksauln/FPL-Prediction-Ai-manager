"""GPT-6 Astra Responses API client for grounded FPL explanations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import httpx
from openai import (
    OpenAI, APIError, APIConnectionError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError,
)

MODEL = "gpt-6-astra"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class LLMError(RuntimeError):
    """User-safe failure without raw provider payloads or credentials."""


@dataclass(frozen=True)
class AstraSettings:
    reasoning_effort: str = "low"
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("Unsupported Astra reasoning effort.")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive.")


def api_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def stream_chat_with_llm(
    *, system_prompt: str, user_prompt: str, settings: AstraSettings,
) -> Iterator[str]:
    """Yield answer text; require completion so partial streams cannot be saved."""
    if not api_key_available():
        raise LLMError("Set OPENAI_API_KEY in the server environment before asking the assistant.")
    try:
        with OpenAI(
            api_key=os.environ["OPENAI_API_KEY"].strip(),
            base_url="https://api.openai.com/v1",
            timeout=settings.timeout_seconds, max_retries=0,
        ) as client:
            with client.responses.create(
                model=MODEL, instructions=system_prompt, input=user_prompt,
                reasoning={"effort": settings.reasoning_effort}, store=False, stream=True,
            ) as stream:
                has_text = False
                completed = False
                for event in stream:
                    if event.type == "response.output_text.delta":
                        has_text = has_text or bool(event.delta.strip())
                        yield event.delta
                    elif event.type == "response.completed":
                        completed = True
                        break
                    elif event.type == "response.incomplete":
                        raise LLMError("The answer is incomplete. Try a shorter question or lower reasoning effort.")
                    elif event.type in ("response.failed", "error"):
                        raise LLMError("The OpenAI request failed. Please retry.")
                    elif event.type == "response.refusal.delta":
                        raise LLMError("The model declined this request. Try rephrasing your FPL question.")
                if not completed:
                    raise LLMError("The response stream ended before completion. Please retry.")
                if not has_text:
                    raise LLMError("The model returned no answer text. Please retry.")
    except AuthenticationError:
        raise LLMError("OpenAI authentication failed. Check the server's OPENAI_API_KEY.") from None
    except (PermissionDeniedError, NotFoundError):
        raise LLMError("GPT-6 Astra is unavailable for this API project. Check model access.") from None
    except RateLimitError:
        raise LLMError("OpenAI rate or quota limit reached. Check billing and retry later.") from None
    except APITimeoutError:
        raise LLMError("OpenAI timed out. Increase the timeout or lower reasoning effort.") from None
    except APIConnectionError:
        raise LLMError("Could not connect to OpenAI. Check the server's network connection.") from None
    except httpx.TransportError:
        raise LLMError("The connection to OpenAI was interrupted. Please retry.") from None
    except APIError:
        raise LLMError("The OpenAI request failed. Please retry.") from None
