import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.nlp.keywords import extract_keywords


class KeywordTests(unittest.TestCase):
    def test_discards_filler_and_keeps_subjects(self):
        words = extract_keywords("ใช่ครับ ระบบธนาคารดีมาก ทีม SA ทำงานกับระบบประกันภัย ระบบช้า")
        for expected in ("ระบบ", "ธนาคาร", "ทีม", "sa", "ประกันภัย"):
            self.assertIn(expected, words)
        for noise in ("ใช่", "ครับ", "ดี", "มาก", "ทำ"):
            self.assertNotIn(noise, words)
        self.assertEqual(words.count("ระบบ"), 1)

    def test_discards_redaction_and_numbers(self):
        words = extract_keywords("ติดต่อ [ถูกปกปิด] 12345 แต่ API ขัดข้อง")
        self.assertNotIn("ถูกปกปิด", words)
        self.assertNotIn("12345", words)
        self.assertIn("api", words)

    def test_discards_combined_opinion_words(self):
        words = extract_keywords("ดีมากครับ งานดีขึ้น แต่ระบบช้า")
        self.assertNotIn("ดีมาก", words)
        self.assertNotIn("ดีขึ้น", words)
        self.assertIn("ระบบ", words)

    def test_keywords_endpoint_keeps_one_result_per_comment(self):
        response = TestClient(app).post(
            "/keywords", json={"texts": ["ระบบช้า ระบบช้า", "ดีมากครับ ทีมช่วย"]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["keywords"][0].count("ระบบ"), 1)
        self.assertIn("ทีม", response.json()["keywords"][1])
        self.assertNotIn("ดีมาก", response.json()["keywords"][1])


if __name__ == "__main__":
    unittest.main()
