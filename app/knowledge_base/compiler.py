import json
from typing import Any

from app.config import LLM_MODEL_NAME
from app.llm.ratelimiter import call_llm_with_backoff


class KnowledgeCompileError(ValueError):
    """Raised when the LLM returns an invalid knowledge article."""


def _clean_string_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item or len(item) > max_len or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def compile_knowledge_base(source: dict, previous_articles: list[dict]) -> dict:
    prompt_payload = {
        "source": source,
        "previous_articles": previous_articles,
    }
    prompt = f"""คุณคือ knowledge compiler ของระบบ MT-Sense สำหรับข้อมูลบรรยากาศองค์กร

หน้าที่ของคุณคือ compile หลักฐานเชิงสถิติที่ผ่านการปกปิดข้อมูลและกฎ n>=5 แล้ว
ให้เป็นบทความฐานความรู้ที่สะสมต่อเนื่องข้ามรอบสำรวจ คล้าย wiki ที่ LLM ดูแลเอง

กฎสำคัญ:
1. ใช้เฉพาะข้อมูลใน INPUT เท่านั้น ห้ามสร้างตัวเลข เหตุการณ์ สาเหตุ หรือข้อสรุปที่ไม่มีหลักฐาน
2. ห้ามกล่าวถึงหรืออนุมานตัวบุคคล พนักงานรายคน หรือข้อความดิบ
3. แยกให้ชัดระหว่าง "ข้อมูลที่เห็น" กับ "ประเด็นที่ควรตรวจสอบต่อ" ห้ามเขียนสมมติฐานเป็นข้อเท็จจริง
4. ถ้า previous_articles มีข้อมูล ให้เชื่อมโยงเฉพาะรอบที่เกี่ยวข้องจริง และใส่ wikilink รูปแบบ [[YYYY-MM]]
5. ตัวเลขทุกตัวในบทความต้องตรงกับ INPUT
6. เขียน summary ภาษาไทยและอังกฤษสั้น กระชับ เหมาะกับ HR/ผู้บริหาร
7. markdown หลักให้เป็นภาษาไทยและมีหัวข้อ: ภาพรวม, สัญญาณสำคัญ, ประเด็นที่ควรติดตาม,
   การเปลี่ยนแปลงจากรอบก่อน (ถ้ามีหลักฐาน), คำถามสำหรับการวิเคราะห์ต่อ, รอบที่เกี่ยวข้อง
8. related_period_ids เลือกได้เฉพาะ period_id ที่อยู่ใน previous_articles
9. tags เป็นรหัส/คำสั้น ๆ ที่ช่วยทำ index ไม่เกิน 8 รายการ
10. suggested_questions เป็นคำถามที่ตอบได้จากฐานความรู้หรือ aggregate ในอนาคต ไม่เกิน 5 ข้อ

INPUT:
{json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)}

ตอบเป็น JSON object เท่านั้น รูปแบบ:
{{
  "title_th": "ชื่อบทความภาษาไทย",
  "title_en": "English title",
  "summary_th": "สรุปภาษาไทย 2-4 ประโยค",
  "summary_en": "English summary in 2-4 sentences",
  "markdown": "# ...",
  "tags": ["workload", "benefits"],
  "related_period_ids": ["period-id-from-input"],
  "suggested_questions": ["คำถาม ..."]
}}
"""

    response = call_llm_with_backoff(prompt, model=LLM_MODEL_NAME)
    try:
        parsed = json.loads(response.text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KnowledgeCompileError("knowledge compiler returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise KnowledgeCompileError("knowledge compiler response must be an object")

    required = ("title_th", "title_en", "summary_th", "summary_en", "markdown")
    cleaned: dict[str, Any] = {}
    for key in required:
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeCompileError(f"knowledge compiler missing {key}")
        cleaned[key] = value.strip()

    if len(cleaned["summary_th"]) > 3000 or len(cleaned["summary_en"]) > 3000:
        raise KnowledgeCompileError("knowledge summary is too long")
    if len(cleaned["markdown"]) > 20000:
        raise KnowledgeCompileError("knowledge article is too long")

    cleaned["tags"] = _clean_string_list(parsed.get("tags"), max_items=8, max_len=64)
    cleaned["related_period_ids"] = _clean_string_list(
        parsed.get("related_period_ids"), max_items=8, max_len=128
    )
    cleaned["suggested_questions"] = _clean_string_list(
        parsed.get("suggested_questions"), max_items=5, max_len=500
    )
    return cleaned
