
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import numpy as np

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Configuration for MongoDB connection
CONNECTION_STRING = "mongodb+srv://tester:dsa4213password@dsa4213-final-project.8fzwk3v.mongodb.net/?retryWrites=true&w=majority&appName=dsa4213-final-project&maxIdleTimeMS=30000"
DB_NAME = "vector_embeddings"
COLLECTION_NAME = "resume_embeddings"

client = MongoClient(CONNECTION_STRING, server_api=ServerApi('1'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_NAME = "resume_embedding_search_index"


def get_embedding(text, embedding_model):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"Error generating embedding for '{text}': {e}")
        return None

# Uses semantic embeddings to find relevant documents   
def semantic_search(query, top_k=2):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = get_embedding(query, embedding_model)
    count = 0
    semantic_results = []
    if query_embedding is None:
        return []
    
    try:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "text": 1,
                    "keywords": 1,
                    "meta": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            },
            { "$sort": { "score": -1 } }
        ])
        for doc in results:
            count += 1
            semantic_results.append(doc)
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []
    print(f"Semantic search completed. Retrieved {count} results.")
    print(f"The first document from semantic search: {semantic_results[0]}")
    # Returns list of dictionaries
    return semantic_results

# Uses keywords to find relevant documents
def keyword_search(keywords, top_k=2):
    compound_should = []
    keyword_results = []
    count = 0
    for entity_type, kw_list in keywords.items():
        for kw in kw_list:
            compound_should.append({
                "text": {
                    "query": kw,
                    "path": f"keywords.{entity_type}",
                    "fuzzy": {
                        "maxEdits": 2,
                        "prefixLength": 1
                    }
                }
            })
    try:
        results = collection.aggregate([
            {
                "$search": {
                    "index": "resume_keyword_search_index",
                    "compound" : {
                        "should" : compound_should,
                        "minimumShouldMatch": 0
                    }
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "text": 1,
                    "keywords": 1,
                    "meta": 1,
                    "score": { "$meta": "searchScore" }
                }
            },
            { "$sort": { "score": -1 } }
        ])
        for doc in results:
            count += 1
            keyword_results.append(doc)
    except Exception as e:
        print(f"Error during keyword search: {e}")
        return []
    print(f"Keyword search completed. Retrieved {count} results.")
    print(f"The first document from keyword search: {keyword_results[0]}")
    # Return list of dictionaries
    return keyword_results

# Using both semantic and keyword search result, rerank them alpha is proportion of semantic search
# Higher alpha = more semantic, less alpha = more keyword
def hybrid_rerank(semantic_results, keyword_results, alpha, top_k):
    # Initialize semantic and keyword search results dataframe
    df_semantic = pd.DataFrame(semantic_results)
    df_keyword = pd.DataFrame(keyword_results)

    if not df_semantic.empty:
        df_semantic['semantic_score'] = df_semantic[['score']]
        df_semantic = df_semantic[['_id', 'text', 'meta', 'semantic_score']]
    if not df_keyword.empty:
        scores = df_keyword[["score"]].to_numpy()
        # Numerically stable softmax (Softmax Normalization)
        exp_scores = np.exp(scores - np.max(scores))
        df_keyword["keyword_score"] = exp_scores / exp_scores.sum()
        df_keyword = df_keyword[['_id', 'text', 'meta', 'keyword_score']]
    # Merge both semantic and keyword search results
    df_merged = pd.merge(df_semantic, df_keyword, on='_id', how='outer')

    df_merged['semantic_score'] = df_merged['semantic_score'].fillna(0)
    df_merged['keyword_score'] = df_merged['keyword_score'].fillna(0)

    # Combining duplicates texts and meta
    df_merged['text'] = df_merged["text_x"].combine_first(df_merged["text_y"])
    df_merged['meta'] = df_merged["meta_x"].combine_first(df_merged["meta_y"])

    # Filter for important columns
    df_merged = df_merged[["_id", "text", "meta", "semantic_score", "keyword_score"]]

    # Get hybrid score
    df_merged['hybrid_score'] = (
        alpha * df_merged['semantic_score'] +
        (1 - alpha) * df_merged['keyword_score']
    )

    # Sort documents based on hybrid_score
    df_merged.sort_values(by='hybrid_score', ascending=False, inplace=True)

    # Return top_k documents
    top_docs = df_merged.head(top_k).to_dict(orient='records')

    return top_docs
    


if __name__ == "__main__":
    # Testing client connection
    try:
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
    except ConnectionFailure as e:
        print("Could not connect to MongoDB:", e)
        exit()

    # Begining semantic query
    semantic_query = "MORTGAGE BANKING FORECLOSURE SPECIALIST Summary Ambitious, self-motivated professional with a passion for quality work. Seeking a baseline opportunity in Underwriting, Lending, Auditing, Quality Assurance, or Analyst roles. Possess large spectrum of experience in the financial industry."
    print(f"Semantic search results for query: '{semantic_query}'")
    results_semantic = semantic_search(semantic_query, top_k=5)

    # Beginning keyword query
    keyword_query = {'SKILL': ['Microsoft Word', 'Power Point'], 'TOOL': ['Oracle']}
    print(f"\nKeyword search results for keywords: {keyword_query}")
    results_keyword = keyword_search(keyword_query, top_k=5)

    results_hybrid = hybrid_rerank(results_semantic, results_keyword, alpha=0.6, top_k=3)
    print(f"hybrid search results: {results_hybrid}")

