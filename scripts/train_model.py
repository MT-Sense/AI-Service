import argparse
import re

import joblib
import pandas as pd
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from app.config import SENTIMENT_MODEL_PATH, VECTORIZER_PATH

TEXT_COLUMN = (
    "ความคิดเห็นและข้อเสนอแนะเกี่ยวกับการทำงาน "
    "(สิ่งที่บริษัท/ทีมทำได้ดี หรือปัญหาหรืออุปสรรคที่พบเจอในขณะนี้)"
)
LABEL_COLUMN = "Sentiment"

BLANK_COMMENT_PATTERN = r'^(-|ไม่มี|ไม่มีครับ|ไม่มีค่ะ|ยังไม่มี|ไม่มีข้อเสนอแนะ|ok|ดีครับ|ดีค่ะ|\.{2,})$'

stopwords = set(thai_stopwords())
important_words = {'ไม่', 'ไม่มี', 'ยัง', 'แต่', 'ดี', 'แย่', 'มาก', 'น้อย'}
stopwords = stopwords - important_words
stopwords.update(['ครับ', 'ค่ะ', 'นะ', 'เลย', 'ๆ', 'ว่า', 'ทำ'])


def process_thai_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'[^ก-๙a-zA-Z\s]', '', text)
    tokens = word_tokenize(text, engine='newmm')
    cleaned_tokens = [word for word in tokens if word not in stopwords and word.strip() != '']

    if len(cleaned_tokens) == 0:
        return "NO_COMMENT"
    return " ".join(cleaned_tokens)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=0)

    is_blank_comment = df[TEXT_COLUMN].str.strip().str.match(BLANK_COMMENT_PATTERN, na=False)
    df_clean = df[~is_blank_comment].dropna(subset=[TEXT_COLUMN]).copy()

    df_clean['cleaned_text'] = df_clean[TEXT_COLUMN].apply(process_thai_text)

    print(f"ข้อมูลทั้งหมด: {len(df)} แถว")
    print(f"หลังกรองข้อความว่าง/ไม่มีสาระ: {len(df_clean)} แถว")

    return df_clean


def train(df_clean: pd.DataFrame):
    X = df_clean['cleaned_text']
    y = df_clean[LABEL_COLUMN]

    vectorizer = TfidfVectorizer()
    X_vector = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vector, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== ผลการประเมินโมเดล ===")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%\n")
    print("Details:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model, vectorizer


def save_model(model, vectorizer):
    joblib.dump(model, SENTIMENT_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("\nบันทึกโมเดลแล้วที่:")
    print(f"   - {SENTIMENT_MODEL_PATH}")
    print(f"   - {VECTORIZER_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment model สำหรับ MT-Sense")
    parser.add_argument(
        "--data",
        required=True,
        help="path ไปยังไฟล์ .xlsx ที่มีข้อมูล feedback (แทนที่ files.upload() ของ Colab)",
    )
    args = parser.parse_args()

    df_clean = load_and_clean_data(args.data)
    model, vectorizer = train(df_clean)
    save_model(model, vectorizer)


if __name__ == "__main__":
    main()