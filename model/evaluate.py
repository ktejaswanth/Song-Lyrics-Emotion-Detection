import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_PATH = "./emotion_model" # Path to your trained model
DATASET_NAME = "dair-ai/emotion"

def evaluate_model():
    # Load model and tokenizer
    try:
        classifier = pipeline("text-classification", model=MODEL_PATH, return_all_scores=False)
        print(f"Loaded model from {MODEL_PATH}")
    except:
        print(f"Could not load model from {MODEL_PATH}. Using default pretrained for demonstration.")
        classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

    # Load test data
    dataset = load_dataset(DATASET_NAME, split="test")
    
    texts = dataset["text"]
    true_labels = dataset["label"]
    
    # Map labels (dataset specific)
    # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    
    print("Running predictions (this may take a while)...")
    preds = classifier(texts, truncation=True)
    
    # Extract predicted labels
    pred_labels = []
    for p in preds:
        # p is like {'label': 'joy', 'score': 0.99}
        # We need to convert label string back to index if strict comparison is needed, 
        # or convert true_labels to strings. Let's convert true_labels to strings.
        pred_labels.append(p['label'])
        
    true_labels_str = [label_names[i] for i in true_labels]
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels_str, pred_labels, labels=label_names)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_labels_str, pred_labels, target_names=label_names))

if __name__ == "__main__":
    evaluate_model()
