import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class AnalyzeBatchContractTest(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.process_pipeline")
    def test_accepts_twelve_comments_in_one_request(self, process_pipeline):
        process_pipeline.return_value = []

        response = self.client.post("/analyze", json={"texts": ["งานเยอะ"] * 12})

        self.assertEqual(response.status_code, 200)
        process_pipeline.assert_called_once()
        self.assertEqual(len(process_pipeline.call_args.kwargs["raw_texts"]), 12)

    @patch("app.main.process_pipeline")
    def test_rejects_more_than_twelve_comments(self, process_pipeline):
        response = self.client.post("/analyze", json={"texts": ["งานเยอะ"] * 13})

        self.assertEqual(response.status_code, 422)
        process_pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
