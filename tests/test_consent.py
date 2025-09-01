import os
import time
import json
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from consent import Config, ConSent
import consent.utils as utils


class TestConSent(unittest.TestCase):

    def setUp(self):
        self.data_df = pd.read_csv(
            "tests/test_data/Chats-EN-ConSent_dummy_data.csv")
        self.data_df = self.data_df.drop(columns=['Unnamed: 0'])
        self.data_df = self.data_df.rename(columns={
            'L1': 'code'}
        )

    def tearDown(self):
        pass

    def test_training_and_inference(self):
        # Split train and test
        train_data_df, test_data_df = \
            utils.train_test_split(self.data_df, test_size=0.5)

        # Define config
        config = Config(**{
            "dataset_name": "Chats-EN-ConSent_dummy_data",
            "code_name": "L1",
            "codes": ["OFF", "COO", "DOM"],
            "default_code": "OFF",
            "language_featurizer": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "sent_hl_units": 10,
            "sent_dropout": 0.5,
            "consent_hl_units": 5,
            "lags": 2,
            "max_epochs": 5,
            "callback_patience": 5,
            "learning_rate": 1e-3,
            "batch_size": 32})

        # Initialize, train, and test
        self.consent = ConSent(config)

        print("\n\nTraining a model with consent.train...\n")
        self.consent.train(train_data_df)
        preds = test_data_df.groupby('dialog_id').apply(
            lambda group: self.consent.predict_sequence(group.name, group),
            include_groups=False
        )

        print("\n\nGenerating predictions using df.groupby().apply()...\n", preds.values)

        # Testing inference with predict_sequence() on yet other dummy data
        preds_dummy = self.consent.predict_sequence('4935ab', pd.DataFrame([
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'hoi'},
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'what we have to do?'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'I think we need to wait'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'or study the first question'},
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'yes what is the frequency?'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'I think 0.5'}]))

        print("\n\nGenerating predictions using consent.predict_sequence()...\n ", preds_dummy)

        pred_message = self.consent.predict_proba(
            dialog_id='4935ab', username='Milhouse', text='do you agree?')
        # pred_message is a tuple of "sent" and "consent" predictions

        print("\n\nGenerating 'sent' and 'consent' predictions using consent.predict_proba()...\n ", pred_message)

    def test_model_saving_and_loading(self):
        # Define config
        config = Config(**{
            "dataset_name": "Chats-EN-ConSent_dummy_data",
            "code_name": "L1",
            "codes": ["OFF", "COO", "DOM"],
            "default_code": "OFF",
            "language_featurizer": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "sent_hl_units": 10,
            "sent_dropout": 0.5,
            "consent_hl_units": 5,
            "lags": 2,
            "max_epochs": 1, # Train for a short period for the test
            "callback_patience": 1,
            "learning_rate": 1e-3,
            "batch_size": 32})

        # Create a temporary directory for saving the model
        temp_model_dir = "./temp_saved_model_test"
        if not os.path.exists(temp_model_dir):
            os.makedirs(temp_model_dir)

        # Initialize and train the model, saving it to the temporary directory
        consent_model = ConSent(config)
        consent_model.train(self.data_df.head(10), save_model=temp_model_dir)

        # Assert that the model directory and .h5 file exist
        self.assertTrue(os.path.isdir(temp_model_dir))
        self.assertTrue(os.path.exists(os.path.join(temp_model_dir, f"{os.path.basename(temp_model_dir)}.h5")))
        self.assertTrue(os.path.exists(os.path.join(temp_model_dir, "config.json")))

        # Load the model from the temporary directory
        loaded_consent_model = ConSent(load=temp_model_dir)

        # Perform a prediction with the loaded model to ensure it works
        pred_message = loaded_consent_model.predict_proba(
            dialog_id='test_dialog', username='test_user', text='This is a test message.')
        self.assertIsNotNone(pred_message)
        self.assertEqual(len(pred_message), 2) # Should return sent_code and consent_code

        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_model_dir)

    @patch('consent.openai_encoder.openai.OpenAI')
    def test_train_with_openai_featurizer(self, mock_openai_class):
        # Mock the OpenAI client and its response
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = np.random.rand(1536).tolist()  # text-embedding-3-small dimension
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Set a dummy API key
        os.environ['OPENAI_API_KEY'] = 'test_key'

        # Define config
        config = Config(**{
            "dataset_name": "Chats-EN-ConSent_dummy_data",
            "code_name": "L1",
            "codes": ["OFF", "COO", "DOM"],
            "default_code": "OFF",
            "language_featurizer": "openai/text-embedding-3-small",
            "sent_hl_units": 10,
            "sent_dropout": 0.5,
            "consent_hl_units": 5,
            "lags": 2,
            "max_epochs": 1,
            "callback_patience": 1,
            "learning_rate": 1e-3,
            "batch_size": 32})

        # Initialize and train
        consent_model = ConSent(config)
        consent_model.train(self.data_df.head(10))

        # Check if the mock was called
        self.assertTrue(mock_client.embeddings.create.called)


if __name__ == '__main__':
    unittest.main()
