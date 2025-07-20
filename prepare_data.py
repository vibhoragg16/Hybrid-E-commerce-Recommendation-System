# prepare_data.py

import pandas as pd
from pathlib import Path
import re

print("Starting advanced data preparation...")

# Define file paths
DATA_PATH = Path("data/")
REVIEWS_FILE = DATA_PATH / "CDs_and_Vinyl.json.gz"
META_FILE = DATA_PATH / "meta_CDs_and_Vinyl.json.gz"
PROCESSED_FILE = DATA_PATH / "processed_reviews_advanced.csv"
USER_VOCAB_FILE = DATA_PATH / "user_vocab.txt"
ITEM_VOCAB_FILE = DATA_PATH / "item_vocab.txt"

# Constants
MIN_INTERACTIONS = 20 

def clean_text(text):
    """A simple function to clean text data."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text) 
    text = text.replace('\n', ' ').strip()
    return text

if not PROCESSED_FILE.exists():
    print(f"Loading raw data...")
    reviews_df = pd.read_json(REVIEWS_FILE, lines=True)
    meta_df = pd.read_json(META_FILE, lines=True)
    
    reviews_df = reviews_df[["reviewerID", "asin", "overall"]]
    meta_df = meta_df[["asin", "title", "description"]]
    
    df = pd.merge(reviews_df, meta_df, on="asin")
    df = df.rename(columns={
        "reviewerID": "user_id",
        "asin": "item_id",
        "title": "item_title",
        "overall": "rating",
        "description": "item_description"
    })
  
    df["item_title"] = df["item_title"].apply(clean_text)
    df["item_description"] = df["item_description"].apply(clean_text)
    df = df[df["item_title"] != ""] # Remove items with no title
    print(f"Initial data loaded with {len(df)} interactions.")

    print(f"Filtering for users/items with at least {MIN_INTERACTIONS} interactions...")
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    valid_items = item_counts[item_counts >= MIN_INTERACTIONS].index
    df_filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
    print(f"Filtered data has {len(df_filtered)} interactions.")
    
    print(f"Saving processed data to {PROCESSED_FILE}...")
    df_filtered.to_csv(PROCESSED_FILE, index=False)
else:
    print(f"Loading pre-processed data from {PROCESSED_FILE}...")
    df_filtered = pd.read_csv(PROCESSED_FILE)

# Create and Save Vocabularies
print("Creating and saving vocabularies...")
user_vocab = df_filtered["user_id"].unique()
item_vocab = df_filtered["item_title"].unique()

with open(USER_VOCAB_FILE, "w", encoding="utf-8") as f:
    for user in user_vocab: f.write(f"{user}\n")

with open(ITEM_VOCAB_FILE, "w", encoding="utf-8") as f:
    for item in item_vocab: f.write(f"{item}\n")

print("-" * 30)
print("Advanced data preparation complete!")
print(f"Total unique users: {len(user_vocab)}")
print(f"Total unique items: {len(item_vocab)}")
print("-" * 30)
