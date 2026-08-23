import json

from google.genai.errors import ClientError

from app.llm.categories import CATEGORIES
from app.llm.ratelimiter import call_llm_with_backoff


def classify_batch_with_llm(items: list[dict]) -> dict:
    items_block = "\n".join(
        f'{{"id": {it["id"]}, "text": "{it["text"]}", '
        f'"sentiment_label": "{it["sentiment_label"]}", '
        f'"sentiment_score": {it["sentiment_score"]:.2f}}}'
        for it in items
    )

    prompt = f"""คุณเป็นระบบจัดหมวดหมู่ความคิดเห็นพนักงานในองค์กร

ต่อไปนี้คือรายการความคิดเห็นของพนักงาน แต่ละรายการมี id, text, และผลวิเคราะห์อารมณ์จากโมเดล ML:

{items_block}

สำหรับแต่ละรายการ จงเลือกหมวดหมู่ที่เกี่ยวข้อง 1 หมวดขึ้นไป จากรายการต่อไปนี้เท่านั้น:
{", ".join(CATEGORIES)}

ตอบกลับเป็น JSON array เท่านั้น ห้ามมีข้อความอื่นนอกเหนือจาก JSON โดยแต่ละรายการต้องมี id ตรงกับที่ให้ไป ในรูปแบบนี้:
[
  {{"id": 0, "categories": ["หมวดที่1"], "reason": "เหตุผลสั้นๆ ไม่เกิน 1 ประโยค"}},
  {{"id": 1, "categories": ["หมวดที่1", "หมวดที่2"], "reason": "เหตุผลสั้นๆ ไม่เกิน 1 ประโยค"}}
]

ตอบให้ครบทุก id ที่ให้ไป ห้ามข้าม ห้ามเพิ่ม id ที่ไม่มีในรายการ
"""

    try:
        response = call_llm_with_backoff(prompt, model="gemini-3.5-flash-lite")
        parsed = json.loads(response.text)
        return {entry["id"]: entry for entry in parsed}
    except json.JSONDecodeError:
        return {it["id"]: {"categories": ["อื่นๆ"], "reason": "parse_error"} for it in items}
    except ClientError:
        return {it["id"]: {"categories": ["อื่นๆ"], "reason": "api_error_after_retries"} for it in items}