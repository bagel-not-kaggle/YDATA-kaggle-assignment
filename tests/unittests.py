import unittest
import pandas as pd
import numpy as np
from preprocess import Preprocessor

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        # Initialize the Preprocessor
        self.preprocessor = Preprocessor()

        # Create a sample dataset
        self.sample_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'campaign_id': [101, 102, 103, 104],
            'product': ['A', 'B', 'C', 'D'],
            'product_category_1': ['cat1', 'cat2', 'cat3', 'cat4'],
            'is_click': [1, 0, 1, 0],
            'session_id': [1001, 1002, 1003, 1004]
        })

    def test_create_user_features(self):
        initial_row_count = len(self.sample_data)
        user_features = self.preprocessor.create_user_features(self.sample_data)
        self.assertIsInstance(user_features, pd.DataFrame)
        self.assertIn('user_id', user_features.columns)
        self.assertIn('campaign_id', user_features.columns)
        self.assertEqual(len(user_features), initial_row_count)

    def test_create_campaign_features(self):
        initial_row_count = len(self.sample_data)
        campaign_features = self.preprocessor.create_campaign_features(self.sample_data)
        self.assertIsInstance(campaign_features, pd.DataFrame)
        self.assertIn('campaign_id', campaign_features.columns)
        self.assertIn('campaign_ctr', campaign_features.columns)
        self.assertEqual(len(campaign_features), initial_row_count)

    def test_create_product_features(self):
        initial_row_count = len(self.sample_data)
        product_features = self.preprocessor.create_product_features(self.sample_data)
        self.assertIsInstance(product_features, pd.DataFrame)
        self.assertIn('product', product_features.columns)
        self.assertIn('product_ctr', product_features.columns)
        self.assertEqual(len(product_features), initial_row_count)

    def test_preprocess_data(self):
        initial_row_count = len(self.sample_data)
        processed_data = self.preprocessor.preprocess_data(self.sample_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIn('user_id', processed_data.columns)
        self.assertIn('campaign_id', processed_data.columns)
        self.assertIn('product', processed_data.columns)
        self.assertLessEqual(len(processed_data), initial_row_count)

if __name__ == '__main__':
    unittest.main()