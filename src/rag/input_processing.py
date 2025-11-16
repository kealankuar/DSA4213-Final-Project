
import os, sys
import json
import pandas as pd
from pathlib import Path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ner.run_inference_fx import run_ner_pipeline
from sentence_transformers import SentenceTransformer
from ner.resume_pdf_parsing import pdf_processing_pipeline
from rag.generate_embeddings import generate_embeddings

input_path_resume_pdf = Path("placeholder").resolve()
RESUME_TEXT_COLUMN = "text"

input_path_query_jsonl = Path("placeholder").resolve()
QUERY_TEXT_COLUMN = "text"

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
NER_MODEL_PATH = str(Path("../../bert-base-cased").resolve())

# Input file: input/queries/query_file/query_file.jsonl
# ner_path: input/queries/query_file/query_file_ner_results.csv
# embedding_path: input/queries/query_file/query_file_ner_results_embedding.jsonl
# NER extract keywords + embeddings for query
def processing_query(input_query_path, embedding_model, ner_model_path, text_column):
    # Convert to Path object if it's a string
    if isinstance(input_query_path, str):
        input_query_path = Path(input_query_path)
    
    # Define intermediate paths
    query_ner_csv_name = input_query_path.stem + "_ner_results.csv"
    query_ner_csv_path = input_query_path.parent / query_ner_csv_name
    query_embedding_jsonl_name = query_ner_csv_path.stem + "_embedding.jsonl"
    query_embedding_jsonl_path = query_ner_csv_path.parent / query_embedding_jsonl_name
    
    # NER extraction
    run_ner_pipeline(
        model_path=ner_model_path,
        input_files=[str(input_query_path)],
        output_dir=str(query_ner_csv_path.parent),
        text_column=text_column,
        conf_threshold=0.55,
        device=-1  # Use CPU instead of GPU
    )
    generate_embeddings(query_ner_csv_path, query_embedding_jsonl_path, text_column, [text_column, 'embedding', 'extracted_entities'], embedding_model)

    return query_embedding_jsonl_path

# Input resume path: input/resume/resume_file/resume_file.pdf
# Input resume_ner_path : input/ner/resume_file/resume_file_ner_results.csv
# Input resume_embedding_path : input/embeddings/resume_file
# 
# 
# 
# 
# /resume_file_ner_results_embedding.jsonl


# NER extract keywords + embeddings for resume
def processing_resume(input_resume_pdf_path, embedding_model, ner_model_path, text_column):
    # Convert to Path object if it's a string
    if isinstance(input_resume_pdf_path, str):
        input_resume_pdf_path = Path(input_resume_pdf_path)
    
    # Define intermediate paths
    input_resume_csv_path = input_resume_pdf_path.with_suffix('.csv')
    input_resume_jsonl_path = input_resume_pdf_path.with_suffix('.jsonl')
    resume_ner_csv_name = input_resume_pdf_path.stem + "_ner_results.csv"
    resume_ner_csv_path = input_resume_pdf_path.parent / resume_ner_csv_name
    resume_embedding_jsonl_name = resume_ner_csv_path.stem + "_embedding.jsonl"
    resume_embedding_jsonl_path = resume_ner_csv_path.parent / resume_embedding_jsonl_name

    # Convert PDF resumes to CSV + JSONL
    pdf_processing_pipeline(input_resume_pdf_path.parent, input_resume_csv_path, input_resume_jsonl_path)

    # NER extraction from JSONL to CSV
    run_ner_pipeline(
        model_path=ner_model_path,
        input_files=[str(input_resume_jsonl_path)],
        output_dir=str(resume_ner_csv_path.parent),
        text_column=text_column,
        conf_threshold=0.55,
        device=-1  # Use CPU instead of GPU
    )

    # Generate embeddings
    generate_embeddings(resume_ner_csv_path, resume_embedding_jsonl_path, text_column, [text_column, 'embedding', 'extracted_entities'], embedding_model)
    return resume_embedding_jsonl_path


if __name__ == "__main__":
    print("Processing query data for NER and embeddings...")
    processing_query(input_path_query_jsonl, EMBEDDING_MODEL, NER_MODEL_PATH, "text")
    print("Query data processing completed.\n")

    print("Processing resume data for NER and embeddings...")
    processing_resume(input_path_resume_pdf, EMBEDDING_MODEL, NER_MODEL_PATH, "text")
    print("Resume data processing completed.")