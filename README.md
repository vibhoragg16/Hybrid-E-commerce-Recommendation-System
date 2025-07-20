# Hybrid E-commerce Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg) ![%F0%9F%A4%97 Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20hugging%20face-Transformers-yellow.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg) ![ScaNN](https://img.shields.io/badge/Index-ScaNN-red.svg)

A scalable, multi-task recommendation engine designed to provide personalized product suggestions for e-commerce platforms. This project leverages a state-of-the-art hybrid architecture to learn from both user behavior and rich product details, deployed as a high-performance API.

---

## 🚀 Key Features

This isn't just a simple collaborative filter; it's an end-to-end system incorporating modern deep learning techniques to build a robust and scalable recommender.

* **Hybrid Two-Tower Architecture:** Utilizes a two-tower neural network to create deep, learned embeddings for both users and products, capturing complex relationships.
* **Transformer-Based Feature Extraction:** Employs a pre-trained **DistilBERT** model from Hugging Face to process raw text from product descriptions, generating rich, context-aware embeddings that go far beyond simple product IDs.
* **Multi-Task Learning:** The model is trained on two objectives simultaneously:
    1.  **Retrieval:** Finding a broad set of relevant products for a user from the entire catalog.
    2.  **Ranking:** Predicting a user's potential rating for an item to fine-tune the final recommendations.
* **Scalable Serving with ScaNN:** Implements Google's **ScaNN (Scalable Nearest Neighbors)** for the final recommendation index. This provides highly efficient, approximate nearest neighbor search, ensuring low-latency performance even with millions of items.
* **Live API Deployment:** The trained model is exposed via a **FastAPI** application, making it easy to integrate with other services and demonstrating a full, end-to-end deployment pipeline.

---

## 🏗️ System Architecture

The model is composed of two main towers that learn embeddings, which are then used for both retrieval and ranking tasks.

```
┌────────────────────────┐                    ┌──────────────────────────────────┐
│       User Tower       │                    │            Item Tower            │
├────────────────────────┤                    ├──────────────────────────────────┤
│      User ID Input     │                    │  Product Description Text Input  │
│           │            │                    │                 │                │
│     StringLookup       │                    │      DistilBERT Tokenizer        │
│           │            │                    │                 │                │
│      Embedding         │                    │        TFDistilBertModel         │
│           │            │                    │                 │                │
└───────────┬────────────┘                    │   GlobalAveragePooling / Dense   │
            │                                 └─────────────────┬────────────────┘
            │                                                   │
            ▼                                                   ▼
┌────────────────────────┐                    ┌──────────────────────────────────┐
│     User Embedding     │                    │          Item Embedding          │
└────────────────────────┘                    └──────────────────────────────────┘
            │                                                   │
            └───────────────────────┬───────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
      ┌───────────────────┐               ┌───────────────────┐
      │  Retrieval Task   │               │   Ranking Task    │
      │ (Find Candidates) │               │ (Predict Rating)  │
      └───────────────────┘               └───────────────────┘
```

---

## 🛠️ Tech Stack

* **Modeling:** TensorFlow, TensorFlow Recommenders, Hugging Face Transformers
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **API Serving:** FastAPI, Uvicorn
* **Indexing:** ScaNN (Scalable Nearest Neighbors)

---

## ⚙️ Setup and Usage

Follow these steps to set up the environment and run the project.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/ecommerce-recommender.git](https://github.com/your-username/ecommerce-recommender.git)
cd ecommerce-recommender
```

### 2. Set Up the Environment

It is recommended to use a Python virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Data

This project uses the "CDs and Vinyl" dataset from the Amazon Review Data (2018) collection.

* **Create the data directory:**
    ```bash
    mkdir data
    ```
* **Download the files:**
    1.  Go to the [official dataset page](https://nijianmo.github.io/amazon/index.html).
    2.  Download `CDs_and_Vinyl.json.gz` (reviews) and `meta_CDs_and_Vinyl.json.gz` (metadata).
    3.  Place both downloaded files into the `data/` folder.

### 4. Run the Pipeline

Execute the scripts in the following order:

* **Step A: Prepare and Preprocess Data**
    This script cleans the raw data, filters it, and saves the processed files and vocabularies.
    ```bash
    python prepare_data.py
    ```

* **Step B: Train the Model**
    This script trains the multi-task Transformer model and builds the scalable ScaNN index. **This process is computationally intensive and will take time.**
    ```bash
    python train.py
    ```

* **Step C: Launch the API Server**
    This command starts the FastAPI server, loading the trained ScaNN index into memory.
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

---

## 🌐 API Endpoints

Once the server is running, you can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Get Recommendations

* **Endpoint:** `/recommendations/{user_id}`
* **Method:** `GET`
* **Description:** Returns the top 10 recommended product titles for a given `user_id`.

**Example Request (`curl`):**

```bash
curl -X GET "[http://127.0.0.1:8000/recommendations/A2M44R4616FR3](http://127.0.0.1:8000/recommendations/A2M44R4616FR3)"
```

**Example JSON Response:**

```json
{
  "user_id": "A2M44R4616FR3",
  "recommendations": [
    "The Best of The Alan Parsons Project",
    "Time Passages",
    "The Turn of a Friendly Card",
    "Eye In The Sky",
    "Ammonia Avenue",
    "Stereotomy",
    "Tales of Mystery and Imagination",
    "Vulture Culture",
    "I Robot",
    "Pyramid"
  ]
}
```

---

## 📂 Project Structure

```
ecommerce-recommender/
├── data/                 # Raw and processed data (ignored by git)
├── saved_model/          # Trained ScaNN index (ignored by git)
├── prepare_data.py       # Script for data loading and preprocessing
├── train.py              # Script for model definition and training
├── main.py               # FastAPI application for serving
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

---

## ✨ Future Improvements

* **Incorporate More Features:** Enhance the user and item towers with additional metadata (e.g., product price, brand, user demographics).
* **Cloud Deployment:** Package the application in a Docker container and deploy it to a cloud service like AWS, GCP, or Azure for a truly production-level setup.
* **A/B Testing Framework:** Implement a framework to test different model versions or ranking strategies in a live environment.
