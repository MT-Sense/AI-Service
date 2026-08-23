import re

from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

stopwords = set(thai_stopwords())
important_words = {'ไม่', 'ไม่มี', 'ยัง', 'แต่', 'ดี', 'แย่', 'มาก', 'น้อย', 'มี'}
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