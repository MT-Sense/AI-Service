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

สำหรับแต่ละรายการ จงเลือกหมวดหมู่หลักที่เกี่ยวข้องจากรหัสต่อไปนี้ ถ้าไม่เกี่ยวข้องกับหมวดใดให้ตอบ []:
{category_list}

ถ้าความคิดเห็นกล่าวถึงประเด็นสำคัญที่ไม่อยู่ในหมวดหลัก ให้เพิ่ม emerging_topics ได้ไม่เกิน 2 หัวข้อ
- ใช้หัวข้อกว้างที่นำไปดำเนินการได้ ไม่ใช้คำอารมณ์ คำฟุ่มเฟือย ชื่อบุคคล หรือข้อมูลส่วนบุคคล
- label_th ยาว 2-40 ตัวอักษร และ label_en เป็นคำแปลภาษาอังกฤษสั้น ๆ
- ความคิดเห็นที่กล่าวถึงเรื่องเดียวกันในชุดนี้ต้องใช้ชื่อหัวข้อเดียวกัน
- ถ้าหมวดหลักครอบคลุมครบแล้วให้ตอบ emerging_topics เป็น []

ตอบกลับเป็น JSON array เท่านั้น ห้ามมีข้อความอื่นนอกเหนือจาก JSON โดยแต่ละรายการต้องมี id ตรงกับที่ให้ไป ในรูปแบบนี้:
[
  {{"id": 0, "categories": ["work"], "emerging_topics": [], "reason": "เหตุผลสั้นๆ ไม่เกิน 1 ประโยค"}},
  {{"id": 1, "categories": [], "emerging_topics": [{{"label_th": "คุณภาพเครื่องมือภายใน", "label_en": "Internal tool quality"}}], "reason": "เหตุผลสั้นๆ ไม่เกิน 1 ประโยค"}}
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
            emerging_topics = entry.get("emerging_topics", [])
            if not isinstance(emerging_topics, list):
                emerging_topics = []
            cleaned_emerging = []
            seen_labels = set()
            for topic in emerging_topics[:2]:
                if not isinstance(topic, dict):
                    continue
                label_th = " ".join(str(topic.get("label_th", "")).split())
                label_en = " ".join(str(topic.get("label_en", "")).split())
                normalized = label_th.casefold()
                if not 2 <= len(label_th) <= 40 or normalized in seen_labels:
                    continue
                seen_labels.add(normalized)
                cleaned_emerging.append({
                    "label_th": label_th,
                    "label_en": label_en[:80] or label_th,
                })
            result[entry["id"]] = {
                "categories": list(dict.fromkeys(
                    category for category in categories
                    if isinstance(category, str) and category in CATEGORIES
                )),
                "emerging_topics": cleaned_emerging,
                "reason": str(entry.get("reason", "")),
            }
        return result
    except (json.JSONDecodeError, ValueError):
        return {it["id"]: {"categories": [], "emerging_topics": [], "reason": "parse_error"} for it in items}
    except ClientError:
        return {it["id"]: {"categories": [], "emerging_topics": [], "reason": "api_error_after_retries"} for it in items}
