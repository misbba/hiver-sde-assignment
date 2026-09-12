import os
import json
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.intent_classifier import TrivialMajorityClassifier, SimpleMLClassifier, IntentClassifier
from src.config import get_eval_set_path, RESULTS_PATH, RANDOM_SEED

def run_full_intent_evaluation(golden_path: str = None) -> Dict:
    """Evaluates all three intent models on verified golden set using 5-fold CV."""
    if golden_path is None:
        golden_path = get_eval_set_path()
        
    df = pd.read_csv(golden_path)
    X = df['message'].tolist()
    y = df['true_intent'].tolist()
    labels = sorted(list(set(y)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    maj_preds, sim_preds, our_preds, y_true_all = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

        # 1. Trivial Majority Baseline
        maj_model = TrivialMajorityClassifier().fit(X_train, y_train)
        maj_preds.extend(maj_model.predict(X_test))

        # 2. Simple ML Baseline
        sim_model = SimpleMLClassifier().fit(X_train, y_train)
        sim_preds.extend(sim_model.predict(X_test))

        # 3. Our AI System Classifier
        our_model = IntentClassifier().fit(X_train, y_train)
        our_preds.extend(our_model.predict(X_test))

        y_true_all.extend(y_test)

    def calc_metrics(y_true, y_pred):
        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(float(p), 4),
            "recall_macro": round(float(r), 4),
            "f1_macro": round(float(f1), 4),
            "confusion_matrix": cm,
            "labels": labels
        }

    results = {
        "dataset_info": {
            "eval_dataset_file": os.path.basename(golden_path),
            "golden_set_size": len(df),
            "num_classes": len(labels),
            "classes": labels
        },
        "models": {
            "trivial_majority_baseline": {
                "name": "Trivial Majority Baseline",
                **calc_metrics(y_true_all, maj_preds)
            },
            "simple_ml_baseline": {
                "name": "TF-IDF + Logistic Regression (Simple ML)",
                **calc_metrics(y_true_all, sim_preds)
            },
            "our_ai_intent_system": {
                "name": "Our AI System (TF-IDF + Calibrated Classifier)",
                **calc_metrics(y_true_all, our_preds)
            }
        }
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    res = run_full_intent_evaluation()
    print("=" * 65)
    print("         AUTOMATED INTENT CLASSIFICATION EVALUATION")
    print("=" * 65)
    print(f"Evaluation Dataset: {res['dataset_info']['eval_dataset_file']} ({res['dataset_info']['golden_set_size']} samples)")
    print("-" * 65)
    for key, model_data in res["models"].items():
        print(f"Model Name:        {model_data['name']}")
        print(f"  Accuracy:        {model_data['accuracy']:.4f}")
        print(f"  Precision (Mac): {model_data['precision_macro']:.4f}")
        print(f"  Recall (Mac):    {model_data['recall_macro']:.4f}")
        print(f"  F1-Score (Mac):  {model_data['f1_macro']:.4f}")
        print("-" * 65)
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 65)
