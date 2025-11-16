import os
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast
from pathlib import Path

# Path configurations
os.chdir(os.path.dirname(os.path.abspath(__file__)))
input_path_resume = Path("../../results/resume_test_data_ner_results.csv").resolve()
output_path_resume = Path("../../data/embeddings/resume_embeddings.jsonl").resolve()

input_path_job = Path("../../results/JD_ner_results.csv").resolve()
output_path_job = Path("../../data/embeddings/job_embeddings.jsonl").resolve() 

input_path_course = Path("../../results/course_data_ner_results.csv").resolve()
output_path_course = Path("../../data/embeddings/course_embeddings.jsonl").resolve()

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# Creating embeddings in dataframe
# Converts csv file to jsonl with embeddings
def generate_embeddings(input_path, output_path, text_column, columns,embedding_model):
    documents = pd.read_csv(input_path).to_dict(orient='records')
    count = 0
    if output_path.parent.exists():
        print(f"Output directory {output_path.parent} already exists.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if documents is None:
            print("No documents provided for embedding generation.")
        else:
            for document in documents:
                text = document.get(text_column, '')
                if text == ' ':
                    print("Skipping empty text.")
                else:
                    embedding = embedding_model.encode(text).tolist()
                    document['embedding'] = embedding
                    filtered_resume = {k:v for k,v in document.items() if k in columns}
                    f.write(json.dumps(filtered_resume, ensure_ascii=False) + '\n')
                    count += 1
    print(f"Embeddings generation of {count} documents completed and saved to: {output_path}")



if __name__ == "__main__":
    print("Generating resume embeddings...")
    generate_embeddings(input_path_resume, output_path_resume, "text", ['text', 'embedding', 'extracted_entities'], EMBEDDING_MODEL)
    print("Resume embeddings generation completed.")

    print("Generating job descroiption embeddings...")
    generate_embeddings(input_path_job, output_path_job, "text", ['text', 'embedding', 'extracted_entities'], EMBEDDING_MODEL)
    print("Job description embeddings generation completed.")

    print("Generating course embeddings...")
    generate_embeddings(input_path_course, output_path_course, "description", ['title', 'url', 'description', 'embedding', 'category', 'sub_category', 'extracted_entities'], EMBEDDING_MODEL)
    print("Course embeddings generation completed.")

    print("All embeddings generation processes completed.")



