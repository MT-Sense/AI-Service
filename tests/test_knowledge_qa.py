import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.knowledge_base.qa import KnowledgeQAError, answer_question
from app.main import app


class KnowledgeQATest(unittest.TestCase):
    @patch("app.knowledge_base.qa.call_llm_with_backoff")
    def test_uses_only_allowed_period_ids(self, call_llm):
        call_llm.return_value = SimpleNamespace(text=json.dumps({
            "answer": "คะแนนเฉลี่ยรอบ 2026-09 คือ 3.8/5",
            "used_period_ids": ["sep", "not-allowed", "sep"],
        }, ensure_ascii=False))

        result = answer_question("คะแนนล่าสุดเท่าไร", [{
            "period_id": "sep",
            "period_label": "2026-09",
            "summary": "ภาพรวม",
            "source_snapshot": {"average_satisfaction": 3.8, "total_responses": 20},
        }])

        self.assertEqual(result["used_period_ids"], ["sep"])
        prompt = call_llm.call_args.args[0]
        self.assertIn("source_snapshot เท่านั้น", prompt)
        self.assertIn("ห้ามอ้างหรืออนุมานความคิดเห็นดิบ", prompt)

    def test_empty_articles_returns_safe_no_evidence_answer(self):
        result = answer_question("เกิดอะไรขึ้น", [], "th")
        self.assertEqual(result["used_period_ids"], [])
        self.assertIn("ข้อมูล", result["answer"])

    def test_rejects_blank_question(self):
        with self.assertRaises(KnowledgeQAError):
            answer_question("   ", [])


class KnowledgeQAEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.answer_question")
    def test_endpoint(self, answer):
        answer.return_value = {"answer": "คำตอบ", "used_period_ids": ["sep"]}
        response = self.client.post("/knowledge/ask", json={
            "question": "ถามอะไรดี",
            "locale": "th",
            "articles": [{"period_id": "sep"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["used_period_ids"], ["sep"])


if __name__ == "__main__":
    unittest.main()
