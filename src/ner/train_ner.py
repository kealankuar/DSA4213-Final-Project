import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate

# --- 1. Configuration ---
MODEL_CHECKPOINT = "distilbert-base-cased" # A fast and effective model
INPUT_CSV = "../../data/ner/ner_training_data_final.csv"
OUTPUT_DIR = "ner_career_model"

# --- 2. Load and Prepare the Dataset ---

# Load the parsed data
print("Loading and preparing the dataset...")
df = pd.read_csv(INPUT_CSV)

# Get all unique tags and create mappings
unique_tags = df['tag'].unique().tolist()
label2id = {tag: i for i, tag in enumerate(unique_tags)}
id2label = {i: tag for i, tag in enumerate(unique_tags)}

# Group tokens and tags by sentence_id
grouped = df.groupby('sentence_id').agg({
    'token': list,
    'tag': lambda x: [label2id[tag] for tag in x]
}).rename(columns={'token': 'tokens', 'tag': 'ner_tags'})

# Convert the pandas DataFrame to a Hugging Face Dataset
hf_dataset = Dataset.from_pandas(grouped)

# Split the dataset into training and testing sets (90% train, 10% test)
train_test_split = hf_dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

print("Dataset prepared:")
print(dataset)

# --- 3. Tokenization and Label Alignment ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    """
    Tokenizes text and aligns labels with the new subword tokens.
    The tokenizer might split a word into multiple subwords. We must align the labels.
    - The first subword gets the original label.
    - Subsequent subwords of the same original word are labeled with -100, which is ignored by the loss function.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Special token (e.g., [CLS], [SEP])
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # First token of a new word
            else:
                label_ids.append(-100) # Subsequent token of the same word
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("\nTokenizing and aligning labels...")
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Data collator handles padding for batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- 4. Metrics for Evaluation ---

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    """
    Computes precision, recall, F1, and accuracy for the NER task.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 5. Model Training ---

print("\nSetting up the model and trainer...")
# Load the pre-trained model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(unique_tags),
    id2label=id2label,
    label2id=label2id
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",  # This should now work correctly
    save_strategy="epoch",      # This should also work correctly
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

# --- 6. Final Evaluation ---
print("\nTraining complete. Evaluating on the test set...")
eval_results = trainer.evaluate()
print("\n--- Final Evaluation Results ---")
print(eval_results)

# Save the final model
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print(f"\n✅ Model saved to '{OUTPUT_DIR}/final_model'")