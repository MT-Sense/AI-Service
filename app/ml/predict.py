import joblib

from app.config import SENTIMENT_MODEL_PATH, VECTORIZER_PATH
from app.nlp.text_cleaning import process_thai_text

loaded_model = joblib.load(SENTIMENT_MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)


def predict_employee_feedback(text: str) -> dict:
    cleaned_input = process_thai_text(text)
    vectorized_input = loaded_vectorizer.transform([cleaned_input])

    prediction = loaded_model.predict(vectorized_input)[0]
    probabilities = loaded_model.predict_proba(vectorized_input)[0]
    classes = loaded_model.classes_

    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    sentiment_score = prob_dict.get("pos", 0) - prob_dict.get("neg", 0)

    return {
        "text": text,
        "cleaned_text": cleaned_input,
        "sentiment_label": prediction,
        "sentiment_score": sentiment_score,
        "confidence": max(prob_dict.values()),
        "probabilities": prob_dict,
    }