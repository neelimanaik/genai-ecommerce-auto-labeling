# src/evaluator.py
import sklearn
import sklearn.metrics
from sklearn.metrics import accuracy_score
from src.classifiers.llm_classifier import classify_zero_shot, classify_few_shot
import json
from src.classifiers.hybrid_classifier import classify_hybrid
print("sklearn works")

def evaluate_hybrid(gold_examples, examples=None):
    predictions, labels = [], []

    for gold_ex in json.loads(gold_examples):
        pred = classify_hybrid(
            gold_ex["Product Description"],
            examples
        )
        predictions.append(pred.strip().lower())
        labels.append(gold_ex["Category"].strip().lower())

    return accuracy_score(labels, predictions)

def evaluate(classify_fn, gold_examples, examples=None):
    """
    Return the accuracy score for predictions on gold examples.
    For each example, we make a prediction using the prompt. Gold labels and
    model predictions are aggregated into lists and compared to compute the
    accuracy.

    Args:
        classify_fn (function): Function to classify product description
        gold_examples (str): JSON string with list of gold examples
        few_examples (str): JSON string with list of few-shot examples
    Output:
        accuracy (float): Accuracy computed by comparing model predictions
                                with ground truth
    """
    predictions, labels = [], []
    SAFE_FALLBACK = "Manual Review"   # or "Hair Care" / "Manual Review"
    for gold_ex in json.loads(gold_examples):
        if examples:
            pred = classify_fn(gold_ex["Product Description"],examples)
        else:
            pred = classify_fn(gold_ex["Product Description"])

        if pred is None:
            pred = SAFE_FALLBACK
            
        predictions.append(pred.strip().lower())
        labels.append(gold_ex["Category"].strip().lower())

    return accuracy_score(labels, predictions)
