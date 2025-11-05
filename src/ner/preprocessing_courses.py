
import os, sys
import json
import pandas as pd
from pathlib import Path

print("Debug: Starting script")
# Change working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Debug: Before sys.path.append")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Debug: Before importing run_inference_fx")
from ner.run_inference_fx import run_ner_pipeline
print("Debug: After importing run_inference_fx")

# Path configurations
output_jsonl_path = Path("../../data/courses/course_data.jsonl").resolve()
input_csv_path = Path("../../data/courses/Online_Courses.csv").resolve()

input_jsonl_path = output_jsonl_path
output_csv_dir = Path("../../results/").resolve()
NER_MODEL_PATH = str(Path("../../bert-base-cased").resolve())

# Creates jsonl file from course csv
def prepare_course_jsonl(input_csv_path, output_path):
    course_df = pd.read_csv(input_csv_path)
    course_df = course_df[['Title', 'URL', 'Short Intro', 'Category', 'Sub-Category', 'Skills']]
    course_df.columns = ['title', 'url', 'description', 'category', 'sub_category', 'skills']
    course_df = course_df.dropna()
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        if course_df is None:
            print("No courses provided for JSONL generation.")
        else:
            for _, course in course_df.iterrows():
                course_dict = course.to_dict()
                f.write(json.dumps(course_dict, ensure_ascii=False) + '\n')
                count += 1
    print(f"Course data JSONL generation of {count} courses completed and saved to: {output_path}")

# Extracts NER entities from course skills and creates a csv file
def extract_ner_entities(input_jsonl_path, ner_model_path):
    run_ner_pipeline(
        model_path=ner_model_path,
        input_files=[str(input_jsonl_path)],
        output_dir=str(output_csv_dir),
        text_column="skills",
        conf_threshold=0.55,
        device=-1  # Use CPU instead of GPU
    )

# Converts course.csv to course_data.jsonl and extracts NER entities into course_data_ner_results.csv
print("Debug: Before __main__ check")
if __name__ == "__main__":
    print("Debug: Inside __main__")
    print(f"Preparing course JSONL from {input_csv_path}")
    prepare_course_jsonl(input_csv_path, output_jsonl_path)
    print(f"Extracted course JSONL saved to: {output_jsonl_path}\n")

    print(f"Extracting NER entities")
    extract_ner_entities(input_jsonl_path, NER_MODEL_PATH)
    print(f"NER extraction completed. Results saved to: {output_csv_dir}")
