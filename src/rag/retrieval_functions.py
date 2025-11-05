import numpy as np
from neo4j import GraphDatabase

# ----------------------------
# Utility: Softmax normalization
# ----------------------------
def softmax(scores, beta=0.5):
    """Applies softmax normalization to a list of keyword (Lucene) scores."""
    scores = np.array(scores, dtype=np.float32)
    exp_scores = np.exp(beta * (scores - np.max(scores)))  # stability trick
    return exp_scores / np.sum(exp_scores)


# ----------------------------
# Main retrieval function
# ----------------------------
def hybrid_retrieve(driver, query_text, embedding_vector, label="JOB", top_k=10, beta=0.5, alpha=0.7):
    """
    Hybrid retrieval combining keyword and semantic search.
    - Softmax normalization is used for Lucene scores.
    - Final score = alpha * semantic + (1 - alpha) * keyword_softmax
    """

    with driver.session() as session:
        # 1️⃣ Keyword (Lucene) search
        lucene_query = f"""
        CALL db.index.fulltext.queryNodes('{label}_text_index', $query_text) 
        YIELD node, score
        RETURN elementId(node) AS id, node.text AS text, score AS keyword_score
        LIMIT {top_k * 5}  // oversample for reranking
        """
        keyword_results = session.run(lucene_query, {"query_text": query_text}).data()

        if not keyword_results:
            print("⚠️ No keyword matches found.")
            return []

        # Extract node ids and keyword scores
        node_ids = [r["id"] for r in keyword_results]
        keyword_scores = [r["keyword_score"] for r in keyword_results]
        keyword_softmax = softmax(keyword_scores, beta=beta)

        # Attach softmax scores
        for i, r in enumerate(keyword_results):
            r["keyword_softmax"] = float(keyword_softmax[i])

        # 2️⃣ Semantic similarity (vector) search
        # (Assumes you stored embeddings as node.embedding)
        semantic_query = f"""
        MATCH (n:{label})
        WHERE elementId(n) IN $node_ids
        WITH n, gds.similarity.cosine(n.embedding, $embedding_vector) AS semantic_score
        RETURN elementId(n) AS id, semantic_score
        """
        semantic_results = session.run(semantic_query, {
            "node_ids": node_ids,
            "embedding_vector": embedding_vector
        }).data()

        # Merge semantic + keyword
        semantic_map = {r["id"]: r["semantic_score"] for r in semantic_results}

        combined = []
        for r in keyword_results:
            semantic_score = semantic_map.get(r["id"], 0.0)
            combined_score = alpha * semantic_score + (1 - alpha) * r["keyword_softmax"]
            combined.append({
                "id": r["id"],
                "text": r["text"],
                "semantic_score": semantic_score,
                "keyword_score": r["keyword_score"],
                "keyword_softmax": r["keyword_softmax"],
                "combined_score": combined_score
            })

        # 3️⃣ Rerank & return
        combined = sorted(combined, key=lambda x: x["combined_score"], reverse=True)
        return combined[:top_k]
