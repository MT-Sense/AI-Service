# ฝึก sentiment model ใน Google Colab

โค้ดนี้ทำขั้นตอนเดียวกับ `scripts/train_model.py` สำหรับไฟล์แบบสอบถามที่มีคอลัมน์ความคิดเห็นและ `Sentiment` (`pos`, `neg`, `neu`; ค่าเก่า `nau` จะถูกแปลงเป็น `neu`) วางใน Colab หนึ่งเซลล์แล้วรัน:

```python
!pip -q install "pythainlp>=5.0.0" "pandas>=2.0.0" "openpyxl>=3.1.0" "scikit-learn==1.6.1" "joblib>=1.3.0"

import re
import joblib
import pandas as pd
from google.colab import files
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_excel(filename, sheet_name=0)

text_column = (
    'ความคิดเห็นและข้อเสนอแนะเกี่ยวกับการทำงาน '
    '(สิ่งที่บริษัท/ทีมทำได้ดี หรือปัญหาหรืออุปสรรคที่พบเจอในขณะนี้)'
)
label_column = 'Sentiment'
blank_patterns = r'^(-|ไม่มี|ไม่มีครับ|ไม่มีค่ะ|ยังไม่มี|ไม่มีข้อเสนอแนะ|\.{2,})$'

is_blank_comment = df[text_column].str.strip().str.match(blank_patterns, na=False)
df_clean = df[~is_blank_comment].dropna(subset=[text_column]).copy()

stopwords = set(thai_stopwords())
important_words = {'ไม่', 'ไม่มี', 'ยัง', 'แต่', 'ดี', 'แย่', 'มาก', 'น้อย', 'มี'}
stopwords -= important_words
stopwords.update(['ครับ', 'ค่ะ', 'นะ', 'เลย', 'ๆ', 'ว่า', 'ทำ'])

def process_thai_text(text):
    text = str(text).strip()
    text = re.sub(r'[^ก-๙a-zA-Z\s]', '', text)
    tokens = word_tokenize(text, engine='newmm')
    cleaned_tokens = [word for word in tokens if word not in stopwords and word.strip() != '']
    return ' '.join(cleaned_tokens) if cleaned_tokens else 'NO_COMMENT'

def canonical_label(value):
    label = str(value).strip().lower()
    if label == 'nau':
        label = 'neu'
    if label not in {'pos', 'neg', 'neu'}:
        raise ValueError(f'unknown sentiment label: {value!r}')
    return label

df_clean['cleaned_text'] = df_clean[text_column].apply(process_thai_text)
data = df_clean[['cleaned_text', label_column]].copy()
data['label'] = data[label_column].map(canonical_label)

no_comment_count = data['cleaned_text'].eq('NO_COMMENT').sum()
data = data[data['cleaned_text'].ne('NO_COMMENT')]
label_counts = data.groupby('cleaned_text')['label'].nunique()
conflicting_texts = label_counts[label_counts > 1].index
conflict_count = data['cleaned_text'].isin(conflicting_texts).sum()
data = data[~data['cleaned_text'].isin(conflicting_texts)]
duplicate_count = data.duplicated(subset='cleaned_text').sum()
data = data.drop_duplicates(subset='cleaned_text').copy()

print(f'ข้อมูลทั้งหมด: {len(df)} แถว')
print(f'หลังกรองข้อความว่าง/ไม่มีสาระ: {len(df_clean)} แถว')
print(f'ตัด NO_COMMENT: {no_comment_count} แถว')
print(f'ตัดข้อความที่มี label ขัดแย้งกัน: {conflict_count} แถว')
print(f'ตัดข้อความซ้ำ: {duplicate_count} แถว')
print(f'ข้อความไม่ซ้ำที่ใช้ฝึกและประเมิน: {len(data)} แถว')

X = data['cleaned_text']
y = data['label']
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError('แต่ละ sentiment ต้องมีอย่างน้อย 2 ข้อความที่ไม่ซ้ำกัน')

# แบ่งก่อนสร้าง TF-IDF เพื่อไม่ให้ข้อมูลในชุดทดสอบรั่วเข้าชุดฝึก
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
vectorizer = TfidfVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vector, y_train)
predictions = model.predict(X_test_vector)

print('Accuracy:', round(accuracy_score(y_test, predictions) * 100, 2), '%')
print('Details:')
print(classification_report(y_test, predictions, zero_division=0))

# ประเมินเสร็จแล้ว ฝึกโมเดลที่จะนำไปใช้จริงด้วยข้อมูลที่ใช้ได้ทั้งหมด
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vector, y)
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
```

หากใช้ไฟล์ที่เพิ่มข้อมูล 217 แถวและเปลี่ยน `nau` เป็น `neu` แล้ว ผลประเมินควรอยู่ใกล้ 69.44% สำหรับชุดข้อมูลและเวอร์ชันไลบรารีเดียวกัน การรันใน Colab จะบันทึกโมเดลใน Colab เท่านั้น; ปุ่มบนเว็บฝึกและเปิดใช้โมเดลใน AI-Service โดยตรง
