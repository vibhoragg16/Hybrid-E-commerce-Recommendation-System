# train_advanced.py

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from pathlib import Path
from transformers import DistilBertTokenizer, TFDistilBertModel

print("Starting ADVANCED model training...")

DATA_PATH = Path("data/")
PROCESSED_FILE = DATA_PATH / "processed_reviews_advanced.csv"
USER_VOCAB_FILE = DATA_PATH / "user_vocab.txt"
ITEM_VOCAB_FILE = DATA_PATH / "item_vocab.txt"
SAVED_MODEL_PATH = Path("saved_model_advanced/recommender_v2")

BERT_MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 64 

print("Loading data and tokenizer...")
df = pd.read_csv(PROCESSED_FILE).astype(str)
tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

with open(USER_VOCAB_FILE, "r", encoding="utf-8") as f:
    user_vocab = f.read().splitlines()
with open(ITEM_VOCAB_FILE, "r", encoding="utf-8") as f:
    item_vocab = f.read().splitlines()

def preprocess_data(row):
    # Tokenize text description for BERT
    bert_inputs = tokenizer(
        row["item_description"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="tf"
    )
    return {
        "user_id": row["user_id"],
        "item_title": row["item_title"],
        "input_ids": tf.squeeze(bert_inputs["input_ids"], axis=0),
        "attention_mask": tf.squeeze(bert_inputs["attention_mask"], axis=0),
        "rating": tf.strings.to_number(row["rating"], out_type=tf.float32)
    }

ratings_ds = tf.data.Dataset.from_tensor_slices(dict(df)).map(preprocess_data)
all_items_ds = tf.data.Dataset.from_tensor_slices(
    dict(df[["item_title", "item_description"]].drop_duplicates())
).map(lambda x: { "item_title": x["item_title"], "item_description": x["item_description"] })

class RecommenderV2(tfrs.Model):
    def __init__(self, user_vocab, item_vocab):
        super().__init__()
        # User Tower
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(user_vocab) + 1, EMBEDDING_DIM)
        ])
        
        # Item Tower (with BERT)
        self.bert_model = TFDistilBertModel.from_pretrained(BERT_MODEL_NAME)
        self.bert_model.trainable = False  
        self.item_model = tf.keras.Sequential([
            self.bert_model,
            tf.keras.layers.GlobalAveragePooling1D(), 
            tf.keras.layers.Dense(EMBEDDING_DIM)     
        ])

        self.ranking_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        self.retrieval_task = tfrs.tasks.Retrieval()
        self.ranking_task = tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def call(self, features):
        user_embedding = self.user_model(features["user_id"])
        item_embedding = self.item_model({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]})[0]
        return (
            user_embedding,
            item_embedding,
            self.ranking_model(tf.concat([user_embedding, item_embedding], axis=1))
        )

    def compute_loss(self, features, training=False):
        user_embeddings, item_embeddings, rating_predictions = self(features)
        
        retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings)
        ranking_loss = self.ranking_task(labels=features["rating"], predictions=rating_predictions)

        return retrieval_loss + ranking_loss

print("Defining and training the advanced model...")
tf.random.set_seed(42)
shuffled = ratings_ds.shuffle(len(df), seed=42, reshuffle_each_iteration=False)
train = shuffled.take(int(len(df) * 0.8)).batch(128).cache() 
test = shuffled.skip(int(len(df) * 0.8)).batch(256).cache()

model = RecommenderV2(user_vocab, item_vocab)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05))
model.fit(train, epochs=2, verbose=1) 

print(f"Creating and saving the ScaNN index to {SAVED_MODEL_PATH}...")

scann = tfrs.layers.factorized_top_k.ScaNN(model.user_model, k=50)

def item_ds_map(features):
    return (
        features["item_title"],
        model.item_model({
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"]
        })[0]
    )
    
all_items_tokenized = all_items_ds.map(preprocess_data).batch(128)
scann.index_from_dataset(all_items_tokenized.map(item_ds_map))

tf.saved_model.save(scann, str(SAVED_MODEL_PATH))

print("-" * 30)
print("Advanced model training and saving complete!")
print("-" * 30)
