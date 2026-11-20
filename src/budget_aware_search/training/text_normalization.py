from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_model_text(value: object) -> str:
    """Normalize text according to what is effectively visible to the encoder.

    We deliberately normalize only representation-equivalent differences:
    Unicode compatibility forms, casing, and whitespace. We do not remove
    punctuation or words because those may contain meaningful product details.
    """
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold()
    text = " ".join(text.split())
    return text
