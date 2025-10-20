from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
output_path = Path("../../data/embeddings/resume_embeddings.jsonl").resolve()
input_path = Path("../../results/test_results.csv").resolve()
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

if output_path.parent.exists():
    print(f"Output directory {output_path.parent} already exists.")
else:
    output_path.parent.mkdir(parents=True, exist_ok=True)


# Extracting keywords from resumes in format {"label" : [list of words]}
def extract_keywords(resume):
    keywords = {}
    if "extracted_entities" not in resume:
        print("No extracted entities found in resume.")
    else:
        for entity in ast.literal_eval(resume.get('extracted_entities', [])):
            label = entity.get('entity_group')
            word = entity.get('word')
            if label and word:
                if label not in keywords.keys():
                    keywords[label] = []
                keywords[label].append(word)
    return keywords

def generate_embeddings(INPUT_PATH, OUTPUT_PATH, doc_type, embedding_model):
    resumes = pd.read_csv(INPUT_PATH).to_dict(orient='records')
    count = 0
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        if resumes is None:
            print("No resumes provided for embedding generation.")
        else:
            for resume in resumes:
                text = resume.get('text', '')
                if text == ' ':
                    print("Skipping empty text.")
                else:
                    embedding = embedding_model.encode(text).tolist()
                    resume['embedding'] = embedding
                    keywords = extract_keywords(resume)
                    resume['keywords'] = keywords
                    meta = ast.literal_eval(resume.get('meta', '{}'))
                    meta['document_type'] = doc_type
                    resume['meta'] = meta
                    cols = ["text", "embedding", "keywords", "meta"]
                    filtered_resume = {k:v for k,v in resume.items() if k in cols}
                    f.write(json.dumps(filtered_resume, ensure_ascii=False) + '\n')
                    count += 1
    print(f"Embeddings generation of {count} resumes completed and saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_embeddings(input_path, output_path, "resume", EMBEDDING_MODEL)




