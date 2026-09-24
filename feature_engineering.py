import pandas as pd
import re

# Load the dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Drop the messy overflow columns (mostly empty, small amount of spilled-over text from commas in messages)
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Rename columns to be clearer
df = df.rename(columns={"v1": "label", "v2": "message"})

# Confirm the clean dataset
print("=== Clean data — first 5 rows ===")
print(df.head())

print("\n=== Dataset shape (rows, columns) ===")
print(df.shape)

print("\n=== Label counts ===")
print(df["label"].value_counts())

# Feature 1: does the message contain a URL?
df["has_url"] = df["message"].apply(lambda x: 1 if re.search(r"http|www|\.com|\.in", x) else 0)

print("\n=== has_url feature ===")
print("Total messages with a URL:", df["has_url"].sum())
print(df[df["has_url"] == 1][["message", "has_url"]].head(5))