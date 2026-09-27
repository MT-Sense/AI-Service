import json
from typing import Any

from app.config import LLM_MODEL_NAME
from app.llm.ratelimiter import call_llm_with_backoff


class KnowledgeQAError(ValueError):
    pass


def answer_question(question: str, articles: list[dict], locale: str = "th") -> dict:
    question = str(question).strip()
    if not question:
        raise KnowledgeQAError("question is required")
    if len(question) > 1200:
        raise KnowledgeQAError("question is too long")
    if not articles:
        return {
            "answer": "ยังไม่มีข้อมูลในฐานความรู้เพียงพอสำหรับตอบคำถามนี้"
            if locale == "th"
            else "There is not enough knowledge-base evidence to answer this question.",
            "used_period_ids": [],
        }

    allowed_ids = {
        str(article.get("period_id", "")).strip()
        for article in articles
        if str(article.get("period_id", "")).strip()
    }
    payload: dict[str, Any] = {
        "question": question,
        "locale": locale,
        "articles": articles,
    }

    prompt = f"""คุณเป็นผู้ช่วย Q&A ของ MT-Sense สำหรับ HR และผู้บริหาร

คุณได้รับเฉพาะ Knowledge Base ที่ compile จาก aggregate ซึ่งผ่านกฎความเป็นส่วนตัว n>=5 แล้ว

กฎบังคับ:
1. ตอบจาก EVIDENCE ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกมาสร้างข้อเท็จจริงเกี่ยวกับองค์กร
2. ห้ามสร้างตัวเลข เปอร์เซ็นต์ แนวโน้ม สาเหตุ หรือข้อสรุปที่ไม่มีอยู่ใน EVIDENCE
3. ห้ามอ้างหรืออนุมานความคิดเห็นดิบ บุคคล ชื่อ อีเมล หรือพนักงานรายคน
4. ถ้าหลักฐานไม่พอ ให้บอกตรง ๆ ว่าข้อมูลไม่พอ ห้ามเดา
5. ถ้าพูดถึงตัวเลข ให้ใช้ตัวเลขจาก source_snapshot เท่านั้น; summary ใช้เป็นบริบทเชิงคำอธิบาย ไม่ใช่แหล่งยืนยันตัวเลข
6. used_period_ids เลือกได้เฉพาะ period_id ที่มีใน EVIDENCE
7. ตอบภาษาไทยเมื่อ locale=th และภาษาอังกฤษเมื่อ locale=en
8. คำตอบควรกระชับ อธิบายว่าหลักฐานมาจากรอบใดเมื่อมีข้อมูลหลายรอบ

EVIDENCE:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

ตอบ JSON object เท่านั้น:
{{
  "answer": "คำตอบ",
  "used_period_ids": ["period-id"]
}}
"""

    response = call_llm_with_backoff(prompt, model=LLM_MODEL_NAME)
    try:
        parsed = json.loads(response.text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KnowledgeQAError("Q&A returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise KnowledgeQAError("Q&A response must be an object")
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise KnowledgeQAError("Q&A response is missing answer")

    used: list[str] = []
    raw_used = parsed.get("used_period_ids", [])
    if isinstance(raw_used, list):
        for value in raw_used:
            if isinstance(value, str) and value in allowed_ids and value not in used:
                used.append(value)

    return {"answer": answer.strip(), "used_period_ids": used}
