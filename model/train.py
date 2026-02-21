import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# 1. Setup
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "dair-ai/emotion"
OUTPUT_DIR = "./emotion_model"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 2. Model
    print("Loading model...")
    num_labels = 6 # sadness, joy, love, anger, fear, surprise
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )
    
    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    
    # 5. Train
    print("Starting training...")
    trainer.train()
    
    # 6. Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Results: {eval_results}")
    
    # 7. Save
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
