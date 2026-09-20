import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import train_model


class TrainingSplitTest(unittest.TestCase):
    def test_removes_duplicate_and_conflicting_cleaned_text(self):
        data = pd.DataFrame({
            'cleaned_text': ['same text', 'same text', 'conflict text', 'conflict text', 'NO_COMMENT', 'unique text'],
            train_model.LABEL_COLUMN: ['pos', 'pos', 'pos', 'neg', 'neu', 'neu'],
        })

        with contextlib.redirect_stdout(io.StringIO()):
            prepared = train_model.prepare_training_data(data)

        self.assertEqual(prepared['cleaned_text'].tolist(), ['same text', 'unique text'])
        self.assertEqual(prepared['label'].tolist(), ['pos', 'neu'])

    def test_vectorizer_fits_holdout_train_before_full_retrain(self):
        data = pd.DataFrame({
            'cleaned_text': [f'feedbackword{i} sharedtoken' for i in range(30)],
            train_model.LABEL_COLUMN: ['neg', 'neu', 'pos'] * 10,
        })
        fit_inputs = []

        class RecordingVectorizer(TfidfVectorizer):
            def fit_transform(self, raw_documents, y=None):
                documents = list(raw_documents)
                fit_inputs.append(documents)
                return super().fit_transform(documents, y)

        with patch.object(train_model, 'TfidfVectorizer', RecordingVectorizer):
            with contextlib.redirect_stdout(io.StringIO()):
                model, vectorizer = train_model.train(data)

        self.assertEqual([len(docs) for docs in fit_inputs], [24, 30])
        self.assertEqual(set(fit_inputs[1]), set(data['cleaned_text']))
        self.assertTrue(set(fit_inputs[0]).issubset(fit_inputs[1]))
        self.assertEqual(set(model.classes_), {'neg', 'neu', 'pos'})
        self.assertEqual(len(vectorizer.vocabulary_), 31)


if __name__ == '__main__':
    unittest.main()
