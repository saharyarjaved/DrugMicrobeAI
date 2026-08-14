from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

excel_file = list(DATA_DIR.glob("*.xlsx"))[0]

df = pd.read_excel(excel_file)

print("Original Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where Drug or Microbe is missing
df = df.dropna(subset=["Name", "Microbe"])

print("After Cleaning:", df.shape)

# Save cleaned dataset
output_file = OUTPUT_DIR / "cleaned_data.csv"
df.to_csv(output_file, index=False)

print("Saved:", output_file)