import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Remove the unused empty columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Rename columns to be more readable
df = df.rename(columns={"v1": "label", "v2": "message"})

# Show the first 5 rows
print(df.head())

# Show how many spam vs ham messages exist
# print(df.iloc[:, 0].value_counts())
# print(df[df["v1"] == "spam"].head())