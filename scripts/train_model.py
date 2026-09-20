import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, accuracy_score

from app.config import SENTIMENT_MODEL_PATH, VECTORIZER_PATH
from app.ml.labels import canonical_label
from app.nlp.text_cleaning import process_thai_text

TEXT_COLUMN = (
    "ความคิดเห็นและข้อเสนอแนะเกี่ยวกับการทำงาน "
    "(สิ่งที่บริษัท/ทีมทำได้ดี หรือปัญหาหรืออุปสรรคที่พบเจอในขณะนี้)"
)
LABEL_COLUMN = "Sentiment"

BLANK_COMMENT_PATTERN = r'^(-|ไม่มี|ไม่มีครับ|ไม่มีค่ะ|ยังไม่มี|ไม่มีข้อเสนอแนะ|\.{2,})$'

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=0)

    is_blank_comment = df[TEXT_COLUMN].str.strip().str.match(BLANK_COMMENT_PATTERN, na=False)
    df_clean = df[~is_blank_comment].dropna(subset=[TEXT_COLUMN]).copy()

    df_clean['cleaned_text'] = df_clean[TEXT_COLUMN].apply(process_thai_text)

    print(f"ข้อมูลทั้งหมด: {len(df)} แถว")
    print(f"หลังกรองข้อความว่าง/ไม่มีสาระ: {len(df_clean)} แถว")

    return df_clean


def prepare_training_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    data = df_clean[['cleaned_text', LABEL_COLUMN]].copy()
    data['label'] = data[LABEL_COLUMN].map(canonical_label)

    no_comment_count = data['cleaned_text'].eq('NO_COMMENT').sum()
    data = data[data['cleaned_text'].ne('NO_COMMENT')]

    label_counts = data.groupby('cleaned_text')['label'].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    conflict_count = data['cleaned_text'].isin(conflicting_texts).sum()
    data = data[~data['cleaned_text'].isin(conflicting_texts)]
    duplicate_count = data.duplicated(subset='cleaned_text').sum()
    data = data.drop_duplicates(subset='cleaned_text').copy()

    print(f"ตัดข้อความไม่มีสาระหลังตัดคำ: {no_comment_count} แถว")
    print(f"ตัดข้อความที่มีป้ายกำกับขัดแย้งกัน: {conflict_count} แถว")
    print(f"ตัดข้อความซ้ำ: {duplicate_count} แถว")
    print(f"ข้อความไม่ซ้ำที่ใช้ฝึกและประเมิน: {len(data)} แถว")
    return data


def train_with_metrics(df_clean: pd.DataFrame):
    data = prepare_training_data(df_clean)
    X = data['cleaned_text']
    y = data['label']
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError('แต่ละ sentiment ต้องมีอย่างน้อย 2 ข้อความที่ไม่ซ้ำกัน')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)

    model = ComplementNB(alpha=0.5)
    model.fit(X_train_vector, y_train)

    predictions = model.predict(X_test_vector)

    print("\n=== ผลการประเมินโมเดล ===")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%\n")
    print("Details:")
    print(classification_report(y_test, predictions, zero_division=0))

    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    metrics = {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'macroF1': float(report['macro avg']['f1-score']),
        'testRows': len(y_test),
        'trainingRows': len(data),
    }

    vectorizer = TfidfVectorizer()
    X_vector = vectorizer.fit_transform(X)
    # model = MultinomialNB()
    model = ComplementNB(alpha=0.5)
    model.fit(X_vector, y)

    return model, vectorizer, metrics


def train(df_clean: pd.DataFrame):
    model, vectorizer, _ = train_with_metrics(df_clean)
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
