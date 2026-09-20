"""Sentiment labels shared by the trained model, API, and Backend-Service."""

VALID_LABELS = {"pos", "neg", "neu"}


def canonical_label(value: object) -> str:
    label = str(value).strip().lower()
    if label == "nau":
        label = "neu"
    if label not in VALID_LABELS:
        raise ValueError(f"unknown sentiment label: {value!r}")
    return label
