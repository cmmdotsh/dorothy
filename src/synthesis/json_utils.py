"""Shared JSON parsing utilities for LLM responses."""

import json
import re


def extract_json(raw: str) -> str:
    """Extract JSON object from LLM response that may contain extra text.

    Models sometimes wrap JSON in markdown fences, think blocks, or preamble.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    # Strip <think>...</think> blocks (thinking mode), including unclosed blocks
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    # Find the first { ... last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling common model quirks.

    1. Strips markdown fences, think blocks, and preamble text
    2. Fixes unescaped newlines inside JSON string values
    """
    extracted = extract_json(raw)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        # Escape literal newlines/carriage-returns inside quoted string values
        fixed = re.sub(
            r'"((?:[^"\\]|\\.)*)"',
            lambda m: '"' + m.group(1).replace("\r", "\\r").replace("\n", "\\n") + '"',
            extracted,
            flags=re.DOTALL,
        )
        return json.loads(fixed)


def ensure_str(value: object) -> str:
    """Coerce an LLM JSON value to a flat string.

    Models sometimes return a nested dict or list instead of a plain string
    for a field. Flatten it so downstream code always gets str.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [ensure_str(v) for v in value.values()]
        return "\n\n".join(p for p in parts if p)
    if isinstance(value, list):
        parts = [ensure_str(v) for v in value]
        return "\n\n".join(p for p in parts if p)
    return str(value) if value is not None else ""


def is_degenerate(text: str, min_words: int) -> bool:
    """Check if LLM output is degenerate (empty, ellipsis, punctuation-only, etc.)."""
    stripped = re.sub(r'[^\w\s]', '', text).strip()
    if not stripped:
        return True
    return len(stripped.split()) < min_words
