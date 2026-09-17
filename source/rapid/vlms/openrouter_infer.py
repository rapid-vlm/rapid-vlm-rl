"""OpenRouter VLM helpers."""
from __future__ import annotations

import base64
import logging
import os
from typing import List, Union

# httpx/httpcore emit DEBUG messages from worker threads, which causes reentrant
# flush errors when IsaacLab's FileHandler is active
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from openai import OpenAI
import time
from typing import Any, Iterable, cast
try:
    from openai.types.chat import ChatCompletionMessageParam
except Exception:  # fallback for older clients
    ChatCompletionMessageParam = Any  # type: ignore


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY (or OPENAI_API_KEY) in environment for OpenRouter access."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _encode_image_to_data_uri(image: Union[str, bytes]) -> str:
    """Return a data URI for a PNG image. Accepts file path or raw bytes."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            data = f.read()
    else:
        data = image
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _with_timeout(c: OpenAI, timeout: int) -> OpenAI:
    try:
        return c.with_options(timeout=timeout)
    except Exception:
        return c


def _chat_with_retry(client: OpenAI, kwargs: dict, timeout_seconds: int, max_retries: int):
    last_exc = None
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            return _with_timeout(client, timeout_seconds).chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                break
            time.sleep(min(2 * (attempt + 1), 6))
    if os.environ.get("OPENROUTER_DEBUG", "0") == "1" and last_exc is not None:
        print(f"[OpenRouter][ERROR] chat.completions failed: {type(last_exc).__name__}: {last_exc}")
    return None


def openrouter_query_two_images_free_form(
    prompt_header_1: str,
    image1_path: str,
    prompt_header_2: str,
    image2_path: str,
    env_prompt_free_form: str,
    summary_prompt_template: str,
    model: str = "google/gemma-3-12b-it",
    max_tokens: int = 1024,
    timeout_seconds: int = 20,
    max_retries: int = 1,
) -> int:
    """Two-stage query: free-form image analysis followed by a deterministic label.

    Returns: 0 if Image 1 better, 1 if Image 2 better, -1 otherwise.
    """
    client = _get_client()
    timeout_seconds = int(os.environ.get("OPENROUTER_TIMEOUT", str(timeout_seconds)))
    max_retries = int(os.environ.get("OPENROUTER_RETRIES", str(max_retries)))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_header_1},
                {"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(image1_path)}},
                {"type": "text", "text": prompt_header_2},
                {"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(image2_path)}},
                {"type": "text", "text": env_prompt_free_form},
            ],
        }
    ]

    analysis = _chat_with_retry(client, {
        "model": model,
        "messages": cast(Iterable[ChatCompletionMessageParam], messages),
        "max_tokens": max_tokens,
    }, timeout_seconds, max_retries)
    if analysis is None:
        return -1
    try:
        analysis_text = (analysis.choices[0].message.content or "").strip()
    except Exception:
        return -1

    summary_prompt = summary_prompt_template.format(analysis_text)
    summary = _chat_with_retry(client, {
        "model": model,
        "messages": cast(Iterable[ChatCompletionMessageParam], [{"role": "user", "content": summary_prompt}]),
        "max_tokens": 16,
    }, timeout_seconds, max_retries)
    if summary is None:
        return -1
    try:
        text = (summary.choices[0].message.content or "").strip().lower()
    except Exception:
        return -1

    if text.startswith("0") or " 0" in text:
        return 0
    if text.startswith("1") or " 1" in text:
        return 1
    return -1


def openrouter_query_two_images_single_prompt(
    image1_path: str,
    image2_path: str,
    combined_prompt: str,
    model: str = "google/gemma-3-12b-it",
    max_tokens: int = 1024,
    timeout_seconds: int = 20,
    max_retries: int = 1,
) -> int:
    """Single-call query: both images + prompt in one request, returns label directly.

    Returns: 0 if Image 1 better, 1 if Image 2 better, -1 otherwise.
    """
    client = _get_client()
    timeout_seconds = int(os.environ.get("OPENROUTER_TIMEOUT", str(timeout_seconds)))
    max_retries = int(os.environ.get("OPENROUTER_RETRIES", str(max_retries)))

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": combined_prompt},
            {"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(image1_path)}},
            {"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(image2_path)}},
        ],
    }]

    resp = _chat_with_retry(client, {
        "model": model,
        "messages": cast(Iterable[ChatCompletionMessageParam], messages),
        "max_tokens": max_tokens,
    }, timeout_seconds, max_retries)
    if resp is None:
        return -1

    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return -1

    import re as _re
    m = _re.search(r"label\s*:\s*(-1|0|1)\b", text, _re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1

    fallback = _re.findall(r"(?<!\d)(-1|0|1)(?!\d)", text)
    if fallback:
        try:
            return int(fallback[-1])
        except Exception:
            return -1
    return -1
