import os
import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# =========================
# 1. CONFIGURATION
# =========================
MODEL_PATH = "../../models/ner_ablation_results/bert-base-cased_lr0.0001"
INPUT_FILES = [
    "../../data/ner/resume_test_data.jsonl",
    "../../data/ner/JD.jsonl"
]
OUTPUT_DIR = "../../results/"
TEXT_COLUMN = "text"
CONF_THRESHOLD = 0.55  # ignore entities below this confidence


# =========================
# 2. FILE LOADING
# =========================
def load_json_or_jsonl(file_path):
    """Load .jsonl, .json, or multi-JSON files safely into a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Handle JSON Lines
    if file_path.endswith(".jsonl") or "\n{" in content:
        data = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(data)

    # Handle normal JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON in {file_path}: {e}")

    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame([{TEXT_COLUMN: v} for v in data.values()])
    else:
        raise ValueError("Unsupported JSON structure.")


# =========================
# 3. TEXT NORMALIZATION
# =========================
def normalize_input_text(text: str) -> str:
    """Preprocess text to ensure consistent BERT tokenization."""
    if not isinstance(text, str):
        return ""
    # Remove Unicode artifacts (e.g., ï¼​)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Add a leading space for tokenization stability
    if not text.startswith(" "):
        text = " " + text
    return text


# =========================
# 4. ENTITY HELPERS
# =========================
def flatten_entities(entities):
    """Convert list of entities into grouped dictionary by label."""
    grouped = {}
    for ent in entities:
        label = ent["entity_group"]
        grouped.setdefault(label, []).append(ent)
    return grouped


def clean_entities(grouped):
    """Clean entity text and filter out junk or low-confidence tokens."""
    cleaned = {}
    for label, ents in grouped.items():
        fixed = []
        for ent in ents:
            word = ent["word"]
            score = float(ent["score"])

            # Basic cleanup
            word = word.replace("##", "")
            word = re.sub(r"[^A-Za-z0-9&\+\-/\.\s]", "", word)
            word = re.sub(r"\s+", " ", word).strip()

            if not word:
                continue

            # Simple replacements for broken subwords
            replacements = {
                r"\b[Pp]p\b": "App",
                r"\b[Ii]on\b": "Dilution",
                r"\b[Uu]ation\b": "Valuation",
                r"\b[Ff]inance?\b": "Finance",
                r"\b[Ff]inancial\b": "Financial",
            }
            for pat, repl in replacements.items():
                word = re.sub(pat, repl, word)

            # Skip low-quality or nonsense words
            if (
                len(word) <= 2
                or len(word.split()) > 10
                or word.lower() in {"anal", "rch", "ï", "th", "on", "um", "st"}
                or re.fullmatch(r"[A-Za-z]{1,2}", word)
                or score < CONF_THRESHOLD
            ):
                continue

            # Fix capitalization but preserve acronyms
            tokens = []
            for w in word.split():
                if w.isupper() or re.search(r"[&/]", w):
                    tokens.append(w)
                else:
                    tokens.append(w.capitalize())
            word = " ".join(tokens)

            fixed.append({"word": word, "score": score})

        if fixed:
            cleaned[label] = fixed
    return cleaned


def merge_similar_labels(entities):
    """Merge plural or similar label variants."""
    merged = {}
    for label, items in entities.items():
        base = label.rstrip("S") if label.endswith("S") and label[:-1] in entities else label
        merged.setdefault(base, []).extend(items)
    return merged


# =========================
# 5. LOAD MODEL
# =========================
print(f"\n🔹 Loading BERT model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)


# =========================
# 6. INFERENCE LOOP
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_path in INPUT_FILES:
    print(f"\n📂 Processing: {os.path.basename(file_path)}")
    df = load_json_or_jsonl(file_path)

    if TEXT_COLUMN not in df.columns:
        print(f"⚠️ Skipping {file_path}: column '{TEXT_COLUMN}' not found")
        continue

    results = []
    texts = df[TEXT_COLUMN].tolist()

    for text in tqdm(texts, desc=f"Extracting entities from {os.path.basename(file_path)}"):
        if not isinstance(text, str) or not text.strip():
            results.append({})
            continue

        text = normalize_input_text(text)  # 🔧 fix tokenizer alignment

        try:
            raw_ents = ner_pipeline(text)
            grouped = flatten_entities(raw_ents)
            cleaned = clean_entities(grouped)
            merged = merge_similar_labels(cleaned)
            results.append(merged)
        except Exception as e:
            print(f"⚠️ Error on text: {e}")
            results.append({})

    df["extracted_entities"] = results

    # Save both CSV and JSON
    stem = Path(file_path).stem
    csv_file = Path(OUTPUT_DIR) / f"{stem}_ner_results.csv"
    json_file = Path(OUTPUT_DIR) / f"{stem}_ner_results.json"

    df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"✅ Saved results to: {csv_file}")
    print(f"✅ Also exported JSON to: {json_file}")

print("\n🎉 All files processed successfully!")
