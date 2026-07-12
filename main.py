import pandas as pd

# Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge dataset
news = pd.concat([fake, true], ignore_index=True)

# Data Cleaning
print(news.duplicated().sum())

news.drop_duplicates(inplace=True)

print(news.isnull().sum())

news.dropna(inplace=True)

news = news.sample(frac=1, random_state=42)
news.reset_index(drop=True, inplace=True)

news = news[["text", "label"]]

print(news.head())
print(news.shape)