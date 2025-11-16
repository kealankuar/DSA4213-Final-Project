import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pathlib import Path
import os

# Configuration for MongoDB connection
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CONNECTION_STRING = "mongodb+srv://tester:dsa4213password@dsa4213-final-project.8fzwk3v.mongodb.net/?retryWrites=true&w=majority&appName=dsa4213-final-project&maxIdleTimeMS=30000"
DB_NAME = "vector_embeddings"
COLLECTION_NAME = "resume_embeddings"
RESUME_EMBEDDING_PATH = Path("../../data/embeddings/resume_embeddings.jsonl").resolve()


# Establish MongoDB connection
client = MongoClient(CONNECTION_STRING, server_api=ServerApi('1'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

def store_embeddings(file_path, collection):
    """Stores embeddings data into MongoDB collection."""

    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            embedding_data = json.loads(line)
            collection.insert_one(embedding_data)
            count += 1
    print(f"Inserted {count} documents into the collection '{COLLECTION_NAME}' in database '{DB_NAME}'.")

if __name__ == "__main__":
    store_embeddings(RESUME_EMBEDDING_PATH, collection)