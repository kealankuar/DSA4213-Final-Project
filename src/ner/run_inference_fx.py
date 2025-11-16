print("Debug: Starting run_inference_fx imports")
try:
    import os
    import json
    import re
    from pathlib import Path
    from typing import Any, cast
    print("Debug: Before pandas import")
    import pandas as pd
    print("Debug: Before tqdm import")
    from tqdm import tqdm
    print("Debug: Before transformers import")
    try:
        print("Debug: Attempting to import AutoTokenizer...")
        from transformers import AutoTokenizer
        print("Debug: Successfully imported AutoTokenizer")
        
        print("Debug: Attempting to import AutoModelForTokenClassification...")
        from transformers import AutoModelForTokenClassification
        print("Debug: Successfully imported AutoModelForTokenClassification")
        
        print("Debug: Attempting to import pipeline...")
        from transformers import pipeline
        print("Debug: Successfully imported pipeline")
        
        print("Debug: All transformers components imported successfully")
    except ImportError as e:
        print(f"ImportError during transformers import: {str(e)}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error during transformers import: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    print("Debug: All imports completed in run_inference_fx")
except Exception as e:
    print(f"Error during imports: {str(e)}")

# Takes JSON/JSONL input, runs NER extraction, outputs CSV + JSON
def run_ner_pipeline(
    model_path: str,
    input_files: list,
    output_dir: str,
    text_column: str = "text",
    conf_threshold: float = 0.55,
    device: int = -1,  # -1 for CPU, >=0 for GPU index
):
    """
    Run NER extraction on one or more JSON/JSONL files using a fine-tuned BERT model.
    
    Args:
        model_path (str): Path to the pretrained/fine-tuned BERT NER model.
        input_files (list): List of JSON/JSONL file paths containing text data.
        output_dir (str): Directory where results will be saved.
        text_column (str): Column name containing the text to analyze.
        conf_threshold (float): Minimum confidence score for extracted entities.
        device (int): Device index for torch (-1=CPU, 0=first GPU, etc.).
    """
    
    # -------------------------
    # Helper functions
    # -------------------------
    def load_json_or_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

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

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Invalid JSON in {file_path}: {e}")

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([{text_column: v} for v in data.values()])
        else:
            raise ValueError("Unsupported JSON structure.")

    def normalize_input_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text.startswith(" "):
            text = " " + text
        return text

    def flatten_entities(entities):
        grouped = {}
        for ent in entities:
            label = ent["entity_group"]
            grouped.setdefault(label, []).append(ent)
        return grouped

    def clean_entities(grouped):
        cleaned = {}
        for label, ents in grouped.items():
            fixed = []
            for ent in ents:
                word = ent["word"]
                score = float(ent["score"])
                word = word.replace("##", "")
                word = re.sub(r"[^A-Za-z0-9&\+\-/\.\s]", "", word)
                word = re.sub(r"\s+", " ", word).strip()

                if not word:
                    continue

                replacements = {
                    r"\b[Pp]p\b": "App",
                    r"\b[Ii]on\b": "Dilution",
                    r"\b[Uu]ation\b": "Valuation",
                    r"\b[Ff]inance?\b": "Finance",
                    r"\b[Ff]inancial\b": "Financial",
                }
                for pat, repl in replacements.items():
                    word = re.sub(pat, repl, word)

                if (
                    len(word) <= 2
                    or len(word.split()) > 10
                    or word.lower() in {"anal", "rch", "ï", "th", "on", "um", "st"}
                    or re.fullmatch(r"[A-Za-z]{1,2}", word)
                    or score < conf_threshold
                ):
                    continue

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
        merged = {}
        for label, items in entities.items():
            base = label.rstrip("S") if label.endswith("S") and label[:-1] in entities else label
            merged.setdefault(base, []).extend(items)
        return merged
    # -------------------------
    # Load model
    # -------------------------
    print(f"\n🔹 Loading BERT model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    ner_pipeline = cast(Any, pipeline)(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

    # -------------------------
    # -------------------------
    # Process files
    # -------------------------

    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_files:
        print(f"\n📂 Processing: {os.path.basename(file_path)}")
        df = load_json_or_jsonl(file_path)

        if text_column not in df.columns:
            print(f"⚠️ Skipping {file_path}: column '{text_column}' not found")
            continue

        results = []
        texts = df[text_column].tolist()

        for text in tqdm(texts, desc=f"Extracting entities from {os.path.basename(file_path)}"):
            if not isinstance(text, str) or not text.strip():
                results.append({})
                continue

            text = normalize_input_text(text)

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

        stem = Path(file_path).stem
        csv_file = Path(output_dir) / f"{stem}_ner_results.csv"
        json_file = Path(output_dir) / f"{stem}_ner_results.json"

        df.to_csv(csv_file, index=False)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

        print(f"✅ Saved results to: {csv_file}")
        print(f"✅ Also exported JSON to: {json_file}")

    print("\n🎉 All files processed successfully!")

