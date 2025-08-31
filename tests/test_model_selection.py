import unittest
import os
import pandas as pd
from consent.model_selection import Handler
from consent import Config

class TestHandler(unittest.TestCase):

    def setUp(self):
        self.dummy_config_path = "tests/test_data/dummy_config.json"
        self.dummy_data_path = "tests/test_data/dummy_data.csv"

    def test_init_with_config_file_path(self):
        handler = Handler(
            data_file_path=self.dummy_data_path,
            config_file_path=self.dummy_config_path
        )
        self.assertIsInstance(handler.config, Config)
        self.assertIsInstance(handler.data_df, pd.DataFrame)
        self.assertEqual(handler.config.dataset_name, "dummy_dataset")
        self.assertEqual(handler.data_df.shape, (4, 5))
        self.assertIn("dialog_id", handler.data_df.columns)
        self.assertIn("code", handler.data_df.columns)

    def test_init_missing_config_and_load_model(self):
        with self.assertRaises(AssertionError) as cm:
            Handler(data_file_path=self.dummy_data_path)
        self.assertIn("Either provide a `config_file_path` or `load_model`.", str(cm.exception))

    def test_init_invalid_config_file_path(self):
        invalid_path = "tests/test_data/non_existent_config.json"
        with self.assertRaises(FileNotFoundError):
            Handler(
                data_file_path=self.dummy_data_path,
                config_file_path=invalid_path
            )

    def test_init_invalid_data_file_path(self):
        invalid_path = "tests/test_data/non_existent_data.csv"
        with self.assertRaises(FileNotFoundError):
            Handler(
                data_file_path=invalid_path,
                config_file_path=self.dummy_config_path
            )