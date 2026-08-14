from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# Automatically first Excel file load karega
excel_file = list(DATA_DIR.glob("*.xlsx"))[0]

print("=" * 60)
print("Reading:", excel_file.name)
print("=" * 60)

df = pd.read_excel(excel_file)

# First rows
print("\nFIRST 5 ROWS")
print(df.head())

# Shape
print("\nSHAPE")
print(df.shape)

# Columns
print("\nCOLUMNS")
print(df.columns.tolist())

# Info
print("\nDATA INFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate rows
print("\nDUPLICATES")
print(df.duplicated().sum())