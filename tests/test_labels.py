import unittest

from app.ml.labels import canonical_label
from app.ml.predict import predict_employee_feedback


class SentimentLabelContractTest(unittest.TestCase):
    def test_legacy_label_maps_to_backend_enum(self):
        self.assertEqual(canonical_label("nau"), "neu")
        self.assertEqual(canonical_label("neu"), "neu")

    def test_saved_model_returns_neu_for_neutral_example(self):
        result = predict_employee_feedback("ปกติ")
        self.assertEqual(result["sentiment_label"], "neu")
        self.assertIn("neu", result["probabilities"])
        self.assertNotIn("nau", result["probabilities"])


if __name__ == "__main__":
    unittest.main()
