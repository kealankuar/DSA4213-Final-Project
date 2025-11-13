import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
import numpy as np
import evaluate, json, shutil
from pathlib import Path

# --- 1. Experiment configurations ---
MODEL_LIST = ["bert-base-cased", "roberta-base", "distilbert-base-uncased"]
LEARNING_RATES = [2e-5, 5e-5, 1e-4]
EPOCHS = 5
INPUT_CSV = "../../data/ner/ner_training_data_final.csv"
OUTPUT_BASE = "../../models/ner_ablation_results"

# --- 2. Load dataset ---
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)

unique_tags = df['tag'].unique().tolist()
label2id = {tag: i for i, tag in enumerate(unique_tags)}
id2label = {i: tag for i, tag in enumerate(unique_tags)}

grouped = df.groupby('sentence_id').agg({
    'token': list,
    'tag': lambda x: [label2id[tag] for tag in x]
}).rename(columns={'token': 'tokens', 'tag': 'ner_tags'})

hf_dataset = Dataset.from_pandas(grouped)
split = hf_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({'train': split['train'], 'test': split['test']})
print(f"Dataset ready: {dataset}")

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 3. Run all experiments ---
results_list = []

for model_name in MODEL_LIST:
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )
        labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    for lr in LEARNING_RATES:

        # --- Fixed directory name with NO timestamp ---
        OUTPUT_DIR = f"{OUTPUT_BASE}/{model_name.replace('/', '_')}_lr{lr}"

        # Clean old folder if it exists
        if Path(OUTPUT_DIR).exists():
            print(f"⚠️ Removing old directory: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)

        print(f"\n🚀 Training {model_name} @ LR={lr}")

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(unique_tags),
            id2label=id2label,
            label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=lr,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir=f"{OUTPUT_DIR}/logs",
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # --- Train ---
        trainer.train()

        # --- Save best model ONLY ---
        print("💾 Saving final best model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # --- Evaluate ---
        metrics = trainer.evaluate()
        metrics.update({"model": model_name, "lr": lr})
        results_list.append(metrics)

        print(f"✅ Done: {model_name} (LR={lr}) → F1={metrics['eval_f1']:.4f}")

# --- 4. Save summary ---
results_df = pd.DataFrame(results_list)
summary_path = f"{OUTPUT_BASE}/ablation_summary.csv"
results_df.to_csv(summary_path, index=False)

print("\n=== All experiments completed ===")
print(results_df.sort_values(by="eval_f1", ascending=False))
print(f"\nSummary saved to {summary_path}")
