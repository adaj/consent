import os
import time
import json
import unittest
import pandas as pd

from consent import Config, ConSent
import consent.utils as utils


class TestConSent(unittest.TestCase):

    def setUp(self):
        self.data_df = pd.read_csv(\
            "tests/test_data/Chats-EN-ConSent_dummy_data.csv",
            index_col=0)
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
            "batch_size": 512})

        # Initialize, train, and test
        self.consent = ConSent(config)

        print("\n\nTraining a model with consent.train...\n")
        self.consent.train(train_data_df)
        preds = test_data_df\
                    .groupby('dialog_id')\
                    .apply(self.consent.predict_sequence)

        print("\n\nGenerating predictions using df.groupby().apply()...\n", preds.values)

        # Testing inference with predict_sequence() on yet other dummy data
        preds_dummy = self.consent.predict_sequence([
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'hoi'},
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'what we have to do?'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'I think we need to wait'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'or study the first question'},
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'yes what is the frequency?'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'I think 0.5'}])

        print("\n\nGenerating predictions using consent.predict_sequence()...\n ", preds_dummy)

        pred_message = self.consent.predict_proba(
            dialog_id='4935ab', username='Milhouse', text='do you agree?')
        # pred_message is a tuple of "sent" and "consent" predictions

        print("\n\nGenerating 'sent' and 'consent' predictions using consent.predict_proba()...\n ", pred_message)

        

if __name__ == '__main__':
    unittest.main()
