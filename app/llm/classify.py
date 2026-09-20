import json

from google.genai.errors import ClientError

from app.llm.categories import CATEGORIES
from app.llm.ratelimiter import call_llm_with_backoff
from app.config import LLM_MODEL_NAME


def classify_batch_with_llm(items: list[dict]) -> dict:
    # Escape quotes and newlines in employee text before including it in the prompt.
    items_block = json.dumps(items, ensure_ascii=False)
    category_list = "\n".join(f'- {key}: {label}' for key, label in CATEGORIES.items())

    prompt = f"""คุณเป็นระบบจัดหมวดหมู่ความคิดเห็นพนักงานในองค์กร

ต่อไปนี้คือรายการความคิดเห็นของพนักงาน แต่ละรายการมี id, text, และผลวิเคราะห์อารมณ์จากโมเดล ML:

{items_block}

สำหรับแต่ละรายการ จงเลือกหมวดหมู่ที่เกี่ยวข้องจากรหัสต่อไปนี้เท่านั้น ถ้าไม่เกี่ยวข้องกับหมวดใดให้ตอบ []:
{category_list}

ตอบกลับเป็น JSON array เท่านั้น ห้ามมีข้อความอื่นนอกเหนือจาก JSON โดยแต่ละรายการต้องมี id ตรงกับที่ให้ไป ในรูปแบบนี้:
[
  {{"id": 0, "categories": ["work"], "reason": "เหตุผลสั้นๆ ไม่เกิน 1 ประโยค ถ้าไม่มี categories ให้เว้นว่าง"}},
  {{"id": 1, "categories": ["team", "manager"], "reason": "เหตุผลสั้นๆ ไม่เกิน 1 ประโยค ถ้าไม่มี categories ให้เว้นว่าง"}}
]

ตอบให้ครบทุก id ที่ให้ไป ห้ามข้าม ห้ามเพิ่ม id ที่ไม่มีในรายการ
"""

    try:
        response = call_llm_with_backoff(prompt, model=LLM_MODEL_NAME)
        parsed = json.loads(response.text)
        if not isinstance(parsed, list):
            raise ValueError("classification is not a list")
        result = {}
        for entry in parsed:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), int):
                continue
            categories = entry.get("categories", [])
            if not isinstance(categories, list):
                categories = []
            result[entry["id"]] = {
                "categories": list(dict.fromkeys(
                    category for category in categories
                    if isinstance(category, str) and category in CATEGORIES
                )),
                "reason": str(entry.get("reason", "")),
            }
        return result
    except (json.JSONDecodeError, ValueError):
        return {it["id"]: {"categories": [], "reason": "parse_error"} for it in items}
    except ClientError:
        return {it["id"]: {"categories": [], "reason": "api_error_after_retries"} for it in items}
