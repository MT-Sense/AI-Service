import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class TrainingEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.workbook = b'PK\x03\x04sample workbook bytes'
        self.report = {
            'accuracy': 0.7,
            'macroF1': 0.6,
            'testRows': 10,
            'trainingRows': 50,
        }

    def test_rejects_missing_training_token(self):
        with patch('app.main.AI_TRAINING_TOKEN', 'test-secret'):
            response = self.client.post('/train', content=self.workbook)
        self.assertEqual(response.status_code, 403)

    def test_rejects_invalid_workbook_before_training(self):
        with patch('app.main.AI_TRAINING_TOKEN', 'test-secret'):
            response = self.client.post('/train', content=b'not an xlsx', headers={'X-Training-Token': 'test-secret'})
        self.assertEqual(response.status_code, 400)

    def test_authorized_upload_returns_training_report(self):
        with patch('app.main.AI_TRAINING_TOKEN', 'test-secret'):
            with patch('app.main._train_uploaded_workbook', return_value=self.report) as train:
                response = self.client.post('/train', content=self.workbook, headers={'X-Training-Token': 'test-secret'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.report)
        train.assert_called_once_with(self.workbook)


if __name__ == '__main__':
    unittest.main()
