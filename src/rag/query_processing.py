
import os, sys
import json
import pandas as pd
from pathlib import Path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ner.run_inference_fx import run_ner_pipeline
from sentence_transformers import SentenceTransformer

input_path_resume_pdf = Path("").resolve()
output_path_resume_ = Path("").resolve()

input_path_query = Path("").resolve()
output_path_query = Path("").resolve()

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
NER_MODEL_PATH = str(Path("../../bert-base-cased").resolve())

def processing_query(query_path):
    return

def processing_resume(resume_path):
    return