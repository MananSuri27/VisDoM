import os
import json
import re
import string
import argparse
from typing import List
from collections import Counter

def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def word_tokenize(text: str) -> List[str]:
    """Tokenize text into words after normalization."""
    normalized = normalize_answer(text)
    return normalized.split()

def calculate_f1(prediction: List[str], ground_truth: List[str]) -> float:
    """Calculate F1 score between prediction and ground truth tokens."""
    prediction_counter = Counter(prediction)
    ground_truth_counter = Counter(ground_truth)
    
    true_positives = sum((prediction_counter & ground_truth_counter).values())
    false_positives = sum(prediction_counter.values()) - true_positives
    false_negatives = sum(ground_truth_counter.values()) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def get_answers_from_json(data):
    """Extract predicted and ground truth answers from JSON."""
    # Try different possible field names
    predicted = ""
    if "Answer" in data:
        predicted = str(data["Answer"])
    elif "answer" in data:
        predicted = str(data["answer"])
    
    ground_truth = str(data.get("gt_answer", ""))
    
    return predicted, ground_truth

def evaluate_directory(directory_path):
    """Evaluate all JSON files in a directory."""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    total_f1 = 0
    valid_files = 0
    
    print(f"Evaluating {len(json_files)} files...")
    
    for filename in json_files:
        filepath = os.path.join(directory_path, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            predicted, ground_truth = get_answers_from_json(data)
            
            if predicted and ground_truth:
                predicted_tokens = word_tokenize(predicted)
                ground_truth_tokens = word_tokenize(ground_truth)
                f1_score = calculate_f1(predicted_tokens, ground_truth_tokens)
                
                total_f1 += f1_score
                valid_files += 1
                
                print(f"{filename}: {f1_score:.4f}")
            else:
                print(f"{filename}: Missing answer fields")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if valid_files > 0:
        avg_f1 = total_f1 / valid_files
        print(f"\nResults:")
        print(f"Average Word F1: {avg_f1:.4f}")
        print(f"Files processed: {valid_files}/{len(json_files)}")
    else:
        print("No valid files to evaluate")

def main():
    parser = argparse.ArgumentParser(description="Word F1 evaluation for response files")
    parser.add_argument("directory", help="Directory containing JSON response files")
    
    args = parser.parse_args()
    evaluate_directory(args.directory)

if __name__ == "__main__":
    main()