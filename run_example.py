import json
import pandas as pd
from consent import Config, ConSent
import consent.utils as utils

# Load the example config
with open("examples/configs/L1__train.json", 'r') as f:
  config = Config(**json.load(f))

# Initialize
consent = ConSent(config)

# Load some data
data_df = pd.read_csv("tests/test_data/Chats-EN-ConSent_dummy_data.csv", 
                      index_col=0)

# Rename the target column to 'code'
data_df = data_df.rename(columns={"L1": "code"})

# Split train and test sets (for the dummy_data, keep test_size=0.5)
train_data_df, test_data_df = \
  utils.train_test_split(data_df, test_size=0.5)

# Train
consent.train(train_data_df)

# Test
print("Testing on unseen data:")
consent.test(test_data_df)

print("Example run successful!")