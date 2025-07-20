import tensorflow as tf
from fastapi import FastAPI
from pathlib import Path

app = FastAPI(
    title="Advanced Recommender API (V2)",
    description="Serves recommendations from a multi-task, Transformer-based model with a ScaNN index.",
    version="2.0"
)

# Define path to the saved model
SAVED_MODEL_PATH = Path("saved_model_advanced/recommender_v2")

print("Loading saved recommendation model...")
recommender = tf.saved_model.load(str(SAVED_MODEL_PATH))
print("Advanced model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Recommender API!", "docs_url": "/docs"}

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_k: int = 10):
    """
    Get top K product recommendations for a given user ID using the ScaNN index.
    """
    _, titles = recommender(tf.constant([user_id]))
    recommendations = [title.decode("utf-8") for title in titles.numpy()[0, :top_k]]
    
    return { "user_id": user_id, "recommendations": recommendations }
