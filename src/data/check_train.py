import pandas as pd

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Train")
print(train.head())

print("\nTest")
print(test.head())

print("\nTrain Labels")
print(train["Label"].value_counts())

print("\nTest Labels")
print(test["Label"].value_counts())