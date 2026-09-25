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
# Feature 2: does the message contain urgency/scam-trigger words?
urgent_words = ["urgent", "winner", "won", "free", "cash", "prize", "claim", "congratulations", "call now", "text now"]

def check_urgent_words(message):
    message_lower = message.lower()
    for word in urgent_words:
        if word in message_lower:
            return 1
    return 0

df["has_urgent_words"] = df["message"].apply(check_urgent_words)

print("\n=== has_urgent_words feature ===")
print("Total messages with urgent words:", df["has_urgent_words"].sum())
print(df[df["has_urgent_words"] == 1][["message", "has_urgent_words"]].head(5))
# Feature 3: how long is the message?
df["message_length"] = df["message"].apply(len)

print("\n=== message_length feature ===")
print("Average length by label:")
print(df.groupby("label")["message_length"].mean())