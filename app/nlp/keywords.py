"""Extract useful words from redacted survey comments for the HR word cloud."""

import re

from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords


_THAI_WORD = re.compile(r"[ก-๙]+\Z")
_LATIN_WORD = re.compile(r"[a-z]+\Z")
_REDACTION = re.compile(r"\[ถูกปกปิด\]")

# The sentiment model keeps negation and opinion words. A word cloud needs a stricter
# list so common answers such as "ดีครับ" do not crowd out concrete subjects.
_NOISE = set(thai_stopwords()) | {
    "ใช่", "ไม่", "ไม่มี", "ดี", "แย่", "ครับ", "ค่ะ", "คะ", "นะ", "จ้า",
    "มาก", "น้อย", "เลย", "ๆ", "ว่า", "ทำ", "มี", "ได้", "อยาก", "รู้สึก",
    "ผม", "ฉัน", "เรา", "คุณ", "พวก", "บริษัท", "พนักงาน", "เดือน", "ก่อน",
    "ตอนนี้", "ปัจจุบัน", "ค่อนข้าง", "โอเค", "ดีมาก", "ดีขึ้น", "ดีเยี่ยม",
    "ยอดเยี่ยม", "แย่มาก", "ใช่ครับ", "ใช่ค่ะ", "ครับผม", "ok", "yes", "no", "good",
    "bad", "the", "and", "for", "with", "this", "that", "from", "are",
}
_SHORT_DOMAIN_WORDS = {"งาน", "ทีม", "sa", "hr", "qa", "ui", "ux"}


def extract_keywords(text: str) -> list[str]:
    """Return unique normalized keywords, counting a word once per response."""
    text = _REDACTION.sub(" ", str(text))
    seen: set[str] = set()
    words: list[str] = []
    for raw in word_tokenize(text, engine="newmm"):
        word = raw.strip().lower()
        if not word or word in _NOISE:
            continue
        if _THAI_WORD.fullmatch(word):
            if len(word) < 3 and word not in _SHORT_DOMAIN_WORDS:
                continue
        elif _LATIN_WORD.fullmatch(word):
            if len(word) < 3 and word not in _SHORT_DOMAIN_WORDS:
                continue
        else:
            continue
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words
