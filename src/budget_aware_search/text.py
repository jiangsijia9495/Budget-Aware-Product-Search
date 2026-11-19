from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from typing import Any

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

FIELD_LABELS = {
    "product_title": "title",
    "product_brand": "brand",
    "product_color": "color",
    "product_bullet_point": "bullets",
    "product_description": "description",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN without importing pandas.
            return ""
    except Exception:
        pass
    text = html.unescape(str(value))
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def build_product_text(
    row: Mapping[str, Any],
    fields: Iterable[str],
    max_chars: int = 1600,
) -> str:
    parts: list[str] = []
    for field in fields:
        value = clean_text(row.get(field, ""))
        if not value:
            continue
        label = FIELD_LABELS.get(field, field)
        parts.append(f"{label}: {value}")
    text = " [SEP] ".join(parts)
    if max_chars > 0:
        text = text[:max_chars].rstrip()
    return text


def basic_tokenize(text: str) -> list[str]:
    """A deterministic, dependency-free tokenizer for lexical diagnostics."""
    return re.findall(r"(?u)\b[\w-]+\b", clean_text(text).lower())
