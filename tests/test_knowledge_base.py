import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.knowledge_base.compiler import KnowledgeCompileError, compile_knowledge_base
from app.main import app


class KnowledgeCompilerTest(unittest.TestCase):
    @patch("app.knowledge_base.compiler.call_llm_with_backoff")
    def test_compiles_structured_article_and_filters_lists(self, call_llm):
        call_llm.return_value = SimpleNamespace(text=json.dumps({
            "title_th": "สรุปบรรยากาศองค์กร 2026-09",
            "title_en": "Organization climate 2026-09",
            "summary_th": "คะแนนเฉลี่ยอยู่ที่ 3.8/5 และประเด็นงานถูกกล่าวถึงมากที่สุด",
            "summary_en": "Average satisfaction is 3.8/5 and work is the most mentioned topic.",
            "markdown": "# สรุป 2026-09\n\n## ภาพรวม\nคะแนนเฉลี่ย 3.8/5",
            "tags": ["work", "work", "", 123, "benefits"],
            "related_period_ids": ["period-aug", "period-aug", 123],
            "suggested_questions": ["ภาระงานเปลี่ยนจากรอบก่อนอย่างไร?"],
        }, ensure_ascii=False))

        result = compile_knowledge_base(
            {
                "period_id": "period-sep",
                "period_label": "2026-09",
                "total_responses": 20,
                "average_satisfaction": 3.8,
            },
            [{"period_id": "period-aug", "period_label": "2026-08", "summary_th": "รอบก่อน"}],
        )

        self.assertEqual(result["tags"], ["work", "benefits"])
        self.assertEqual(result["related_period_ids"], ["period-aug"])
        prompt = call_llm.call_args.args[0]
        self.assertIn('"total_responses": 20', prompt)
        self.assertIn("[[YYYY-MM]]", prompt)
        self.assertIn("ห้ามกล่าวถึงหรืออนุมานตัวบุคคล", prompt)

    @patch("app.knowledge_base.compiler.call_llm_with_backoff")
    def test_rejects_incomplete_llm_output(self, call_llm):
        call_llm.return_value = SimpleNamespace(text='{"title_th":"missing fields"}')
        with self.assertRaises(KnowledgeCompileError):
            compile_knowledge_base({"period_id": "p"}, [])


class KnowledgeEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.compile_knowledge_base")
    def test_endpoint_returns_compiled_article(self, compile_kb):
        compiled = {
            "title_th": "สรุป",
            "title_en": "Summary",
            "summary_th": "สรุปไทย",
            "summary_en": "English summary",
            "markdown": "# สรุป",
            "tags": ["work"],
            "related_period_ids": [],
            "suggested_questions": ["เกิดอะไรขึ้น?"],
        }
        compile_kb.return_value = compiled

        response = self.client.post("/knowledge/compile", json={
            "source": {"period_id": "period-sep", "total_responses": 12},
            "previous_articles": [],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), compiled)
        compile_kb.assert_called_once()

    @patch("app.main.compile_knowledge_base", side_effect=KnowledgeCompileError("bad output"))
    def test_endpoint_maps_invalid_llm_output_to_502(self, _compile_kb):
        response = self.client.post("/knowledge/compile", json={
            "source": {"period_id": "period-sep"},
            "previous_articles": [],
        })
        self.assertEqual(response.status_code, 502)


if __name__ == "__main__":
    unittest.main()
