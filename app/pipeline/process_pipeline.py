from app.config import BATCH_SIZE, DEFAULT_CONFIDENCE_THRESHOLD
from app.llm.classify import classify_batch_with_llm
from app.ml.predict import predict_employee_feedback


def process_pipeline(
    raw_texts: list[str],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    ml_results = [predict_employee_feedback(text) for text in raw_texts]

    results = []
    for batch_start in range(0, len(ml_results), BATCH_SIZE):
        batch = ml_results[batch_start: batch_start + BATCH_SIZE]

        items = [
            {
                "id": i,
                "text": r["text"],
                "sentiment_label": r["sentiment_label"],
                "sentiment_score": r["sentiment_score"],
            }
            for i, r in enumerate(batch)
        ]

        classification_map = classify_batch_with_llm(items)

        for i, ml_result in enumerate(batch):
            classification = classification_map.get(
                i, {"categories": ["อื่นๆ"], "reason": "missing_from_response"}
            )
            results.append({
                **ml_result,
                "categories": classification["categories"],
                "reason": classification["reason"],
                "low_confidence": ml_result["confidence"] < confidence_threshold,
            })

        print(f"{len(results)}/{len(raw_texts)} ข้อความ")

    return results