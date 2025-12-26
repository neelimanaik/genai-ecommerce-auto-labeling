import pandas as pd
import numpy as np
from src.evaluation.evaluator import evaluate, evaluate_hybrid
from src.classifiers.hybrid_classifier import classify_hybrid
from src.classifiers.llm_classifier import classify_zero_shot, classify_few_shot
import os
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from src.utils.example_sampler import create_examples

print(os.getcwd())

if __name__ == "__main__":
    data = pd.read_csv("F:/Neelima/GEN_AI/src/data/auto-labelling.csv")
    
    examples_df, gold_examples_df = train_test_split(
        data, #<- the full dataset
        test_size=0.8, #<- 80% random sample selected for gold examples
        random_state=42, #<- ensures that the splits are the same for every session
        stratify=data['Category'] #<- ensures equal distribution of labels
    )
    gold_examples = (
        gold_examples_df.to_json(orient='records')
        )
   
    # For each run create a new sample of examples
    examples = create_examples(examples_df)
    hybrid_performance = []

    hybrid_accuracy, hybrid_records = evaluate_hybrid(gold_examples, examples)
    hybrid_performance.append(hybrid_accuracy)
    print("Mean Hybrid accuracy:", np.array(hybrid_performance).mean())
    print("Std Hybrid accuracy:", np.array(hybrid_performance).std())
    
    os.makedirs("outputs", exist_ok=True)

    df = pd.DataFrame(hybrid_records)
    df.to_csv("outputs/hybrid_predictions.csv", index=False)

    print("Saved predictions to outputs/hybrid_predictions.csv")

    few_shot_performance, zero_shot_performance = [], []
    num_eval_runs = 5

    for _ in tqdm(range(num_eval_runs)):

        # For each run create a new sample of examples
        examples = create_examples(examples_df)

        # Evaluate prompt accuracy on gold examples
        few_shot_accuracy, few_shot_records =  evaluate(classify_few_shot, gold_examples,examples)
        zero_shot_accuracy, zero_shot_records =  evaluate(classify_zero_shot, gold_examples)

        few_shot_performance.append(few_shot_accuracy)
        zero_shot_performance.append(zero_shot_accuracy)
    
    print("Few-shot accuracy array:", few_shot_performance)
    print("Zero-shot accuracy array:", zero_shot_performance)

    print("Mean of Few-shot accuracy over", num_eval_runs, "runs:", np.array(few_shot_performance).mean())
    print("Mean of Zero-shot accuracy over", num_eval_runs, "runs:", np.array(zero_shot_performance).mean())

    print("Standard Deviation of Few-shot accuracy over", num_eval_runs, "runs:", np.array(few_shot_performance).std())
    print("Standard Deviation of Zero-shot accuracy over", num_eval_runs, "runs:", np.array(zero_shot_performance).std())

    os.makedirs("outputs", exist_ok=True)

    df = pd.DataFrame(few_shot_records)
    df.to_csv("outputs/few_shot_predictions.csv", index=False)

    print("Saved predictions to outputs/few_shot_predictions.csv")

    os.makedirs("outputs", exist_ok=True)

    df = pd.DataFrame(zero_shot_records)
    df.to_csv("outputs/zero_shot_predictions.csv", index=False)

    print("Saved predictions to outputs/zero_shot_predictions.csv")