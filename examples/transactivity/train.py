import pandas as pd
from consent import ConSent, Config
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score
import os

# Load environment variables
load_dotenv()

# Load data
train_df = pd.read_csv('examples/transactivity/train.csv')
test_df = pd.read_csv('examples/transactivity/test.csv')

# Get the codes from the Transactivity column
codes = train_df['code'].unique().tolist()

# --- Model 1: Universal Sentence Encoder ---

print("--- Training Universal Sentence Encoder model ---")
use_config = Config(**{
    "dataset_name": "transactivity",
    "wandb_project": "transactivity-comparison",
    "code_name": "code",
    "codes": codes,
    "default_code": codes[0],
    "language_featurizer": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
    "sent_hl_units": 512,
    "sent_dropout": 0.3,
    "consent_hl_units": 32,
    "lags": 5,
    "max_epochs": 50,
    "callback_patience": 10,
    "learning_rate": 5e-4,
    "batch_size": 128
})

use_model = ConSent(use_config)
use_model.train(train_df, save_model="examples/transactivity/models/use_model")
kappa_use, f1_use = use_model.test(test_df)
print(f"Cohen's Kappa (USE): {kappa_use}")
print(f"F1 Score (USE): {f1_use}")


# # --- Model 2: OpenAI Embeddings ---

# print("\n--- Training OpenAI Embeddings model ---")
# openai_config = Config(**{
#     "dataset_name": "transactivity",
#     "wandb_project": "transactivity-comparison",
#     "code_name": "code",
#     "codes": codes,
#     "default_code": codes[0],
#     "language_featurizer": "openai/text-embedding-3-small",
#     "sent_hl_units": 512,
#     "sent_dropout": 0.3,
#     "consent_hl_units": 32,
#     "lags": 5,
#     "max_epochs": 100,
#     "callback_patience": 10,
#     "learning_rate": 5e-4,
#     "batch_size": 128
# })

# openai_model = ConSent(openai_config)
# openai_model.train(train_df, save_model="examples/transactivity/models/openai_model")
# kappa_openai, f1_openai = openai_model.test(test_df)

# print(f"Cohen's Kappa (OpenAI): {kappa_openai}")
# print(f"F1 Score (OpenAI): {f1_openai}")

