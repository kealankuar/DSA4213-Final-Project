from graph_retrieval import CareerEngine
from input_processing import processing_query, processing_resume
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
NER_MODEL_PATH = str(Path("../../bert-base-cased").resolve())
RESUME_TEXT_COLUMN = "text"
QUERY_TEXT_COLUMN = "text"

NEO4J_URI="neo4j+s://accc1403.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="TwKAMXkflTo1NnauW3SEjMW4iXbExFIHA6oNi4mN1h4"


def context_generation_function(input_query_path, input_resume_path, uri, user, password):
    # Process query and resume to get their embedding jsonl paths
    query_embedding_jsonl_path = processing_query(
        input_query_path,
        EMBEDDING_MODEL,
        NER_MODEL_PATH,
        text_column=QUERY_TEXT_COLUMN
    )

    resume_embedding_jsonl_path = processing_resume(
        input_resume_path,
        EMBEDDING_MODEL,
        NER_MODEL_PATH,
        text_column=RESUME_TEXT_COLUMN
    )

    # Initialize CareerEngine and get recommendations
    career_engine = CareerEngine(uri, user, password, query_embedding_jsonl_path, resume_embedding_jsonl_path)
    full_context = career_engine.get_full_context()

    return full_context
    