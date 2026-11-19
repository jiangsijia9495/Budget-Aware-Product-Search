from __future__ import annotations

ESCI_GAINS: dict[str, float] = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0.0,
}

# Cross-encoder classification order. Expected ranking gain is computed from this order.
CLASS_TO_ID: dict[str, int] = {"I": 0, "C": 1, "S": 2, "E": 3}
ID_TO_CLASS: dict[int, str] = {value: key for key, value in CLASS_TO_ID.items()}
CLASS_GAINS = [ESCI_GAINS[ID_TO_CLASS[i]] for i in range(len(ID_TO_CLASS))]


def validate_esci_labels(labels: set[str]) -> None:
    unknown = labels.difference(ESCI_GAINS)
    if unknown:
        raise ValueError(f"Unknown ESCI labels: {sorted(unknown)}")
