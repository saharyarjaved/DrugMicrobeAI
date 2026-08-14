import pandas as pd

df = pd.read_csv("data/processed/cleaned_data.csv")

print("Unique Drugs:", df["Name"].nunique())
print("Unique Microbes:", df["Microbe"].nunique())
print("Unique Strains:", df["Strain"].nunique())

print("\nTop 10 Drugs")
print(df["Name"].value_counts().head(10))

print("\nTop 10 Microbes")
print(df["Microbe"].value_counts().head(10))