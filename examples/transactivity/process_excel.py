import pandas as pd

# Load the Excel file
excel_file = 'examples/transactivity/dialogues_transactivity_regulation.xlsx'
xls = pd.ExcelFile(excel_file)

# Create a list to hold the dataframes
dfs = []

# Iterate over the sheets
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name)
    df['dialog_id'] = sheet_name
    dfs.append(df)

# Concatenate all dataframes
all_df = pd.concat(dfs, ignore_index=True)

# Save to a CSV file
all_df.to_csv('examples/transactivity/transactivity_data.csv', index=False)

print("Successfully converted the Excel file to a single CSV file.")
