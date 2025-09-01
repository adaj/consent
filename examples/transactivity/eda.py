import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create EDA directory
eda_dir = 'examples/transactivity/eda'
os.makedirs(eda_dir, exist_ok=True)

# Load the data
df = pd.read_csv('examples/transactivity/transactivity_data.csv')

# Print key stats
print("Data Info:")
df.info()
print("\nValue Counts:")
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())

# --- Visualizations ---

# Class distribution for categorical columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=col, data=df, order = df[col].value_counts().index)
    plt.title(f'Class Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f'{col}_distribution.png'))
    plt.close()

# Distribution of numeric columns per dialog_id
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='dialog_id', y=col, data=df)
    plt.title(f'Distribution of {col} per dialog_id')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f'{col}_distribution_per_dialog.png'))
    plt.close()

print("\nEDA visualizations saved in examples/transactivity/eda")
