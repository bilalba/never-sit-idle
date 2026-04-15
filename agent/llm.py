"""OpenRouter LLM client with retries, token counting, and tool-call parsing."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import requests
import tiktoken

from agent import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return approximate token count for *text*."""
    return len(_get_encoder().encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """Rough token count across a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # role / structural overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += count_tokens(json.dumps(part))
        # tool_calls field
        tc = msg.get("tool_calls")
        if tc:
            total += count_tokens(json.dumps(tc))
    return total


# ---------------------------------------------------------------------------
# API call with retries
# ---------------------------------------------------------------------------

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Non-retryable LLM error."""


class LLMRetryExhausted(Exception):
    """All retry attempts exhausted."""

    def __init__(self, last_error: Exception, attempts: int):
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(f"LLM call failed after {attempts} attempts: {last_error}")


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """Call OpenRouter chat completions with automatic retries.

    Returns the raw JSON response dict from the API.
    Raises LLMRetryExhausted if all retries are spent.
    Raises LLMError for non-retryable failures.
    """
    api_key = api_key or config.OPENROUTER_API_KEY
    model = model or config.OPENROUTER_MODEL
    max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

    is_openrouter = "openrouter.ai" in config.OPENROUTER_BASE_URL

    if is_openrouter and not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set")

    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/never-sit-idle"
        headers["X-Title"] = "never-sit-idle-agent"

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("LLM call attempt %d/%d", attempt, max_retries)
            resp = requests.post(
                config.OPENROUTER_BASE_URL,
                headers=headers,
                json=body,
                timeout=120,
            )

            # Retryable HTTP errors
            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = LLMError(
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )
                _backoff(attempt)
                continue

            # Non-retryable HTTP errors
            if resp.status_code != 200:
                raise LLMError(
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()

            # OpenRouter sometimes wraps errors in 200 responses
            if "error" in data:
                err_msg = data["error"]
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", str(err_msg))
                raise LLMError(f"API error: {err_msg}")

            return data

        except requests.exceptions.Timeout:
            last_error = LLMError("Request timed out")
            _backoff(attempt)
            continue
        except requests.exceptions.ConnectionError as exc:
            last_error = LLMError(f"Connection error: {exc}")
            _backoff(attempt)
            continue
        except LLMError:
            raise
        except Exception as exc:
            last_error = exc
            _backoff(attempt)
            continue

    raise LLMRetryExhausted(last_error, max_retries)  # type: ignore[arg-type]


def chat_completion_stream(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    on_delta: Any | None = None,
) -> dict[str, Any]:
    """Streaming chat completion — calls on_delta(text) for each text chunk.

    Returns the same response dict as chat_completion once the stream ends.
    Falls back to non-streaming if the server doesn't support SSE.
    """
    api_key = api_key or config.OPENROUTER_API_KEY
    model = model or config.OPENROUTER_MODEL
    max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

    is_openrouter = "openrouter.ai" in config.OPENROUTER_BASE_URL

    if is_openrouter and not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/never-sit-idle"
        headers["X-Title"] = "never-sit-idle-agent"

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                config.OPENROUTER_BASE_URL,
                headers=headers,
                json=body,
                timeout=120,
                stream=True,
            )

            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = LLMError(f"HTTP {resp.status_code}")
                _backoff(attempt)
                continue

            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            # Parse SSE stream
            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            role = "assistant"
            finish_reason = None
            usage: dict[str, Any] = {}
            response_id = ""

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                response_id = chunk.get("id", response_id)
                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if "role" in delta:
                    role = delta["role"]

                # Text content
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    if on_delta:
                        on_delta(delta["content"])

                # Tool calls (accumulated across chunks)
                if "tool_calls" in delta:
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.get("id", f"call_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_calls_acc[idx]
                        fn_delta = tc_delta.get("function", {})
                        if "name" in fn_delta:
                            acc["function"]["name"] = fn_delta["name"]
                        if "arguments" in fn_delta:
                            acc["function"]["arguments"] += fn_delta["arguments"]

            # Assemble final response in the same shape as non-streaming
            message: dict[str, Any] = {
                "role": role,
                "content": "".join(content_parts),
            }
            if tool_calls_acc:
                message["tool_calls"] = [
                    tool_calls_acc[i] for i in sorted(tool_calls_acc)
                ]

            return {
                "id": response_id,
                "choices": [{"message": message, "finish_reason": finish_reason}],
                "usage": usage,
            }

        except requests.exceptions.Timeout:
            last_error = LLMError("Request timed out")
            _backoff(attempt)
        except requests.exceptions.ConnectionError as exc:
            last_error = LLMError(f"Connection error: {exc}")
            _backoff(attempt)
        except LLMError:
            raise
        except Exception as exc:
            last_error = exc
            _backoff(attempt)

    raise LLMRetryExhausted(last_error, max_retries)  # type: ignore[arg-type]


def _backoff(attempt: int) -> None:
    delay = (config.RETRY_BACKOFF_BASE ** attempt) + random.uniform(
        0, config.RETRY_JITTER_MAX
    )
    logger.info("Backing off %.1fs before retry", delay)
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """Pull the assistant message out of an OpenRouter response."""
    choices = response.get("choices", [])
    if not choices:
        raise LLMError("No choices in LLM response")
    return choices[0].get("message", {})


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of tool-call dicts from an assistant message.

    Each dict has: {id, name, arguments (already parsed as dict)}.
    Handles both the structured tool_calls format and fallback
    JSON-in-content parsing for models that don't natively support tools.
    """
    calls: list[dict[str, Any]] = []

    # --- Structured tool_calls (OpenAI-compatible) ---
    raw_calls = message.get("tool_calls", [])
    for tc in raw_calls:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        else:
            args = args_raw
        calls.append(
            {
                "id": tc.get("id", f"call_{id(tc)}"),
                "name": fn.get("name", ""),
                "arguments": args,
            }
        )

    # --- Fallback: parse JSON tool calls from content ---
    if not calls:
        content = message.get("content", "")
        if content:
            calls = _parse_tool_calls_from_content(content)

    return calls


def _parse_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Best-effort extraction of tool calls embedded as JSON in content.

    Handles multiple formats:
      1. Standard JSON: {"tool": "name", "arguments": {...}}
      2. JSON array: [{"tool": "name", "arguments": {...}}]
      3. Gemma-style: <|tool_call>call:tool_name{key:value}<tool_call|>
         with <|"|> as string delimiters
    """
    # First try Gemma-style tool calls
    calls = _parse_gemma_tool_calls(content)
    if calls:
        return calls

    # Then try standard JSON
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        calls = _try_parse_json_block(content, start_char, end_char)
        if calls:
            return calls

    return []


def _normalize_gemma_strings(content: str) -> str:
    """Replace <|"|>...<|"|> pairs with properly escaped JSON strings.

    Gemma uses <|"|> as string delimiters.  Any literal " characters inside
    a delimited pair must be escaped to \\" so the result is valid JSON when
    the delimiters are replaced with regular double-quotes.
    """
    DELIM = '<|"|>'
    parts: list[str] = []
    pos = 0

    while True:
        open_idx = content.find(DELIM, pos)
        if open_idx == -1:
            # No more delimiters — append the rest as-is
            parts.append(content[pos:])
            break

        close_idx = content.find(DELIM, open_idx + len(DELIM))
        if close_idx == -1:
            # Unmatched opening delimiter — append the rest as-is
            parts.append(content[pos:])
            break

        # Text before the opening delimiter (unmodified)
        parts.append(content[pos:open_idx])

        # The string value between delimiters — escape for valid JSON
        inner = content[open_idx + len(DELIM):close_idx]
        # json.dumps produces a properly escaped JSON string (with outer quotes)
        parts.append(json.dumps(inner))

        pos = close_idx + len(DELIM)

    return "".join(parts)


def _parse_gemma_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse Gemma's native tool call format.

    Gemma outputs: <|tool_call>call:tool_name{key:<|"|>value<|"|>, ...}<tool_call|>
    The <|"|> tokens are string delimiters used instead of regular quotes.
    """
    calls: list[dict[str, Any]] = []

    # Normalize Gemma string delimiters to regular quotes, escaping any
    # literal " characters that appear inside delimited strings so they
    # don't break JSON parsing.
    normalized = _normalize_gemma_strings(content)

    # Match patterns like: call:tool_name{...}
    import re
    pattern = r'call:(\w+)\{(.+?)\}(?:</?tool_call\|?>|$)'
    for match in re.finditer(pattern, normalized, re.DOTALL):
        name = match.group(1)
        args_str = "{" + match.group(2) + "}"

        # Try to parse as JSON
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            # Try fixing common issues: unquoted keys, trailing commas
            try:
                # Add quotes around unquoted keys
                fixed = re.sub(r'(\w+):', r'"\1":', args_str)
                # Remove trailing commas before }
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                args = json.loads(fixed)
            except json.JSONDecodeError:
                # Last resort: extract key-value pairs manually
                args = _extract_kv_pairs(args_str)

        calls.append({
            "id": f"gemma_{id(match)}",
            "name": name,
            "arguments": args,
        })

    return calls


def _extract_kv_pairs(text: str) -> dict[str, Any]:
    """Best-effort key-value extraction from malformed JSON-like strings."""
    import re
    result: dict[str, Any] = {}
    # Match key:"value" or key:[...] patterns
    for m in re.finditer(r'"?(\w+)"?\s*:\s*("(?:[^"\\]|\\.)*"|\[.*?\]|[^,}\]]+)', text, re.DOTALL):
        key = m.group(1)
        val = m.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            # Use json.loads to properly unescape JSON string escapes (e.g. \" → ")
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                val = val[1:-1]
        elif val.startswith('['):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass
        result[key] = val
    return result


def _try_parse_json_block(
    content: str, start_char: str, end_char: str
) -> list[dict[str, Any]]:
    start = content.find(start_char)
    if start == -1:
        return []

    # Find matching close bracket
    depth = 0
    end = start
    for i, ch in enumerate(content[start:], start):
        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
        if depth == 0:
            end = i
            break
    if depth != 0:
        return []

    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return []

    items = parsed if isinstance(parsed, list) else [parsed]
    calls: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and ("tool" in item or "name" in item):
            name = item.get("tool") or item.get("name", "")
            args = item.get("arguments", item.get("params", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(
                {
                    "id": f"fallback_{id(item)}",
                    "name": name,
                    "arguments": args if isinstance(args, dict) else {},
                }
            )

    return calls
