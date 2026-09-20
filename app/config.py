import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
AI_TRAINING_TOKEN = os.environ.get("AI_TRAINING_TOKEN", "")
LLM_MODEL_NAME = "gemini-3.5-flash-lite"

BATCH_SIZE = 12

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

DEFAULT_CONFIDENCE_THRESHOLD = 0.55
