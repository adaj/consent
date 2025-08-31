import unittest
import os
import pandas as pd
from consent.model_selection import Handler
from consent import Config
from unittest import mock

class TestHandler(unittest.TestCase):

    def setUp(self):
        self.dummy_config_path = "tests/test_data/dummy_config.json"
        self.dummy_data_path = "tests/test_data/dummy_data.csv"
        self.dummy_hyper_config_path = "tests/test_data/dummy_hyper_config.json"

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

    @mock.patch('consent.model_selection.ParameterSampler')
    @mock.patch('consent.model_selection.ConSent')
    @mock.patch('consent.wandb')
    def test_hyperparameter_tuning_parses_experiment_grid(self, mock_wandb, mock_ConSent, mock_ParameterSampler):
        # Setup Handler with hyperparameter config
        handler = Handler(
            data_file_path=self.dummy_data_path,
            config_file_path=self.dummy_hyper_config_path
        )

        # Configure the mock ConSent instance
        # Create a dummy DataFrame that predict_sequence would return
        dummy_predictions_df = pd.DataFrame({
            'username': ['u1'],
            'text': ['t1'],
            'code': ['A'],
            'sent_code': ['A'],
            'consent_code': ['A']
        })
        # Set the return value for the predict_sequence method of the mock ConSent instance
        mock_ConSent.return_value.predict_sequence.return_value = dummy_predictions_df

        # Mock the return value of ParameterSampler to control iterations
        mock_ParameterSampler.return_value = iter([
            {'sent_hl_units': 10, 'sent_dropout': 0.5} # Just one iteration for this test
        ])

        # Call hyperparameter_tuning
        handler.hyperparameter_tuning(n_iter=1, val_size=0.5)

        # Assert that ParameterSampler was called with the correct param_distributions
        expected_param_distributions = {
            'sent_hl_units': [10, 20],
            'sent_dropout': [0.5, 0.6]
        }
        mock_ParameterSampler.assert_called_once_with(
            param_distributions=expected_param_distributions,
            n_iter=1,
            random_state=handler.random_state
        )

    @mock.patch('consent.model_selection.ParameterSampler')
    @mock.patch('consent.model_selection.ConSent')
    @mock.patch('consent.wandb')
    def test_hyperparameter_tuning_n_iter_and_combinations(self, mock_wandb, mock_ConSent, mock_ParameterSampler):
        # Setup Handler with hyperparameter config
        handler = Handler(
            data_file_path=self.dummy_data_path,
            config_file_path=self.dummy_hyper_config_path
        )

        # Configure the mock ConSent instance
        dummy_predictions_df = pd.DataFrame({
            'username': ['u1'], 'text': ['t1'], 'code': ['A'],
            'sent_code': ['A'], 'consent_code': ['A']
        })
        mock_ConSent.return_value.predict_sequence.return_value = dummy_predictions_df

        # Define the expected parameter combinations
        expected_combinations = [
            {'sent_hl_units': 10, 'sent_dropout': 0.5},
            {'sent_hl_units': 10, 'sent_dropout': 0.6},
            {'sent_hl_units': 20, 'sent_dropout': 0.5},
            {'sent_hl_units': 20, 'sent_dropout': 0.6},
        ]
        # Mock the return value of ParameterSampler to control iterations
        mock_ParameterSampler.return_value = iter(expected_combinations)

        # Call hyperparameter_tuning with n_iter=-1 to test all combinations
        handler.hyperparameter_tuning(n_iter=-1, val_size=0.5)

        # Assert that ParameterSampler was called with the correct param_distributions
        expected_param_distributions = {
            'sent_hl_units': [10, 20],
            'sent_dropout': [0.5, 0.6]
        }
        mock_ParameterSampler.assert_called_once_with(
            param_distributions=expected_param_distributions,
            n_iter=len(expected_combinations), # Should be 4 combinations
            random_state=handler.random_state
        )
        # Assert that ConSent.train was called for each combination
        self.assertEqual(mock_ConSent.return_value.train.call_count, len(expected_combinations))
        # Assert that wandb.log was called for each combination
        self.assertEqual(mock_ConSent.return_value.wandb_run.log.call_count, len(expected_combinations))

    @mock.patch('consent.model_selection.ParameterSampler')
    @mock.patch('consent.model_selection.ConSent')
    @mock.patch('consent.wandb')
    @mock.patch('consent.model_selection.Handler.compute_metrics') # Mock compute_metrics method
    @mock.patch('consent.model_selection.utils.train_test_split') # Mock train_test_split globally
    def test_hyperparameter_tuning_calls_train_and_evaluation(self, mock_train_test_split, mock_compute_metrics, mock_wandb, mock_ConSent, mock_ParameterSampler):
        # Define the expected train_data and test_data that would be returned by train_test_split
        expected_train_data = pd.DataFrame({
            'dialog_id': ['dialog_1', 'dialog_1'],
            'username': ['user_a', 'user_b'],
            'text': ['hello', 'hi there'],
            'code': ['A', 'B'],
            'timestamp': ['2023-01-01', '2023-01-01']
        })
        expected_test_data = pd.DataFrame({
            'dialog_id': ['dialog_2', 'dialog_2'],
            'username': ['user_a', 'user_b'],
            'text': ['how are you', 'fine thanks'],
            'code': ['C', 'A'],
            'timestamp': ['2023-01-02', '2023-01-02']
        })
        mock_train_test_split.return_value = (expected_train_data, expected_test_data)

        # Setup Handler with hyperparameter config
        handler = Handler(
            data_file_path=self.dummy_data_path,
            config_file_path=self.dummy_hyper_config_path
        )

        # Configure the mock ConSent instance
        dummy_predictions_df = pd.DataFrame({
            'username': ['u1', 'u2'],
            'text': ['t1', 't2'],
            'code': ['C', 'A'], # These should match the 'code' column of expected_test_data
            'sent_code': ['C', 'A'],
            'consent_code': ['C', 'A']
        })
        mock_ConSent.return_value.predict_sequence.return_value = dummy_predictions_df

        # Mock the return value of ParameterSampler to control iterations
        expected_combinations = [
            {'sent_hl_units': 10, 'sent_dropout': 0.5},
            {'sent_hl_units': 20, 'sent_dropout': 0.6},
        ]
        mock_ParameterSampler.return_value = iter(expected_combinations)

        # Mock the return value of compute_metrics
        mock_compute_metrics.return_value = {'accuracy': 0.8}

        # Call hyperparameter_tuning
        handler.hyperparameter_tuning(n_iter=len(expected_combinations), val_size=0.5)

        # Assert that ConSent.train was called for each combination
        self.assertEqual(mock_ConSent.return_value.train.call_count, len(expected_combinations))
        # Assert that compute_metrics was called for each combination
        self.assertEqual(mock_compute_metrics.call_count, len(expected_combinations))

        # Assert train was called with the mocked train_data
        actual_train_data_call = mock_ConSent.return_value.train.call_args[0][0]
        pd.testing.assert_frame_equal(actual_train_data_call, expected_train_data)
        self.assertEqual(mock_ConSent.return_value.train.call_args[1]['tf_verbosity'], 2) # tf_verbosity

        # Assert compute_metrics was called with the correct arguments
        actual_y_true_call = mock_compute_metrics.call_args[0][0]
        actual_y_pred_call = mock_compute_metrics.call_args[0][1]
        actual_config_call = mock_compute_metrics.call_args[0][2]

        pd.testing.assert_series_equal(actual_y_true_call, expected_test_data['code'])
        pd.testing.assert_series_equal(actual_y_pred_call, dummy_predictions_df['consent_code'])
        self.assertIsInstance(actual_config_call, Config)
        # You might want to assert specific attributes of the config if needed
        self.assertEqual(actual_config_call.dataset_name, "dummy_dataset")