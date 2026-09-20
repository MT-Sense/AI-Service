import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.llm.classify import classify_batch_with_llm


class ClassifyContractTest(unittest.TestCase):
    @patch("app.llm.classify.call_llm_with_backoff")
    def test_returns_dashboard_topic_ids_and_escapes_text(self, call_llm):
        call_llm.return_value = SimpleNamespace(text=json.dumps([
            {"id": 0, "categories": ["work", "team", "unknown", "work"], "reason": "งานในทีม"}
        ]))

        result = classify_batch_with_llm([
            {"id": 0, "text": 'งาน "เร่ง"\nทีม', "sentiment_label": "neg", "sentiment_score": -0.8}
        ])

        self.assertEqual(result[0]["categories"], ["work", "team"])
        prompt = call_llm.call_args.args[0]
        self.assertIn('งาน \\"เร่ง\\"\\nทีม', prompt)


if __name__ == "__main__":
    unittest.main()
