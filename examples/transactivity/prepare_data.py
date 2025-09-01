import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('examples/transactivity/transactivity_data.csv')

# --- Data Cleaning ---

# Rename the column with a trailing space
if 'Transactivity ' in df.columns:
    df.rename(columns={'Transactivity ': 'Transactivity_extra', 'Transcription': 'text', 'Speaker': 'username', 'Transactivity': 'code'}, inplace=True)

# Drop the extra transactivity and regulation columns
df.drop(columns=['Transactivity_extra', 'Regulation'], inplace=True)

# Drop rows with missing values in Transcription or Transactivity
df.dropna(subset=['text', 'code'], inplace=True)

# Clean username column
df['username'] = df['username'].str.strip().str.lower()

# --- Train-Test Split ---

# Get unique dialog_ids
dialog_ids = df['dialog_id'].unique()

# Split dialog_ids into train and test
train_ids, test_ids = train_test_split(dialog_ids, test_size=0.2, random_state=42)

# Create train and test dataframes
train_df = df[df['dialog_id'].isin(train_ids)]
test_df = df[df['dialog_id'].isin(test_ids)]

# Save to CSV files
train_df.to_csv('examples/transactivity/train.csv', index=False)
test_df.to_csv('examples/transactivity/test.csv', index=False)

print("Successfully cleaned the data and created train.csv and test.csv")
