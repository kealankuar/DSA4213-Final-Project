import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import json

# --- 1. Configuration ---
MODEL_PATH = "../../models/ner_career_model/final_model"
INPUT_FILE = "../../data/ner/resume_test_data.jsonl"
OUTPUT_FILE = "../../results/test_results.csv"
TEXT_COLUMN = 'text'

# --- 2. More Robust Data Loading ---
def load_jsonl_robust(file_path):
    """
    Loads a JSONL file line by line, providing clear errors for bad lines.
    """
    data = []
    print(f"Loading and reading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Skip empty lines
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"🚨 Error decoding JSON on line {i + 1}: {e}")
                    print(f"   Problematic Line: {line.strip()}")
                    return None
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"🚨 Error: Input file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"🚨 An unexpected error occurred while reading the file: {e}")
        return None

df = load_jsonl_robust(INPUT_FILE)

if df is None:
    print("Stopping script due to data loading errors.")
    exit()

if TEXT_COLUMN not in df.columns:
    print(f"🚨 Error: The specified TEXT_COLUMN '{TEXT_COLUMN}' was not found.")
    print(f"   Available columns are: {df.columns.tolist()}")
    exit()

# --- 3. Load Model and Run Inference ---
print(f"\nLoading NER model from: {MODEL_PATH}")
try:
    ner_pipeline = pipeline("ner", model=MODEL_PATH, aggregation_strategy="simple", device=0)
except Exception as e:
    print(f"🚨 Error loading model: {e}")
    exit()

texts = df[TEXT_COLUMN].tolist()
print(f"\nRunning NER inference on {len(texts)} resumes...")

results = []
for text in tqdm(texts, desc="Extracting Entities"):
    if isinstance(text, str) and text.strip():
        try:
            entities = ner_pipeline(text)
            results.append(entities)
        except Exception as e:
            print(f"Warning: Could not process text. Error: {e}")
            results.append([]) # Append empty list on error
    else:
        results.append([])

df['extracted_entities'] = results

# --- 4. Display and Save Results ---
print("\n--- Inference Complete. Showing first 5 results: ---")
print(df.head())

df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Successfully saved results to '{OUTPUT_FILE}'")