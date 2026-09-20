import joblib
from threading import RLock

from app.config import SENTIMENT_MODEL_PATH, VECTORIZER_PATH
from app.ml.labels import canonical_label
from app.nlp.text_cleaning import process_thai_text

loaded_model = joblib.load(SENTIMENT_MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)
model_lock = RLock()


def activate_model(model, vectorizer) -> None:
    global loaded_model, loaded_vectorizer
    with model_lock:
        loaded_model = model
        loaded_vectorizer = vectorizer


def predict_employee_feedback(text: str) -> dict:
    cleaned_input = process_thai_text(text)
    with model_lock:
        vectorized_input = loaded_vectorizer.transform([cleaned_input])
        prediction = canonical_label(loaded_model.predict(vectorized_input)[0])
        probabilities = loaded_model.predict_proba(vectorized_input)[0]
        classes = loaded_model.classes_

    prob_dict: dict[str, float] = {}
    for cls, prob in zip(classes, probabilities):
        label = canonical_label(cls)
        prob_dict[label] = prob_dict.get(label, 0.0) + float(prob)
    sentiment_score = prob_dict.get("pos", 0) - prob_dict.get("neg", 0)

    print(f"prediction: {prediction}")
    print(f"sentiment_score: {sentiment_score}")

    return {
        "text": text,
        "cleaned_text": cleaned_input,
        "sentiment_label": prediction,
        "sentiment_score": sentiment_score,
        "confidence": max(prob_dict.values()),
        "probabilities": prob_dict,
    }
