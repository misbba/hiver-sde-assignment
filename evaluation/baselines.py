import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.intent_classifier import TrivialMajorityClassifier, SimpleMLClassifier
from src.config import get_eval_set_path, RANDOM_SEED

def evaluate_baselines(golden_path: str = None) -> Dict:
    """Evaluates Majority Baseline and Simple ML (TF-IDF + Logistic Regression) on verified golden set."""
    if golden_path is None:
        golden_path = get_eval_set_path()
        
    df = pd.read_csv(golden_path)
    X = df['message'].tolist()
    y = df['true_intent'].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    majority_preds, simple_preds, y_true_all = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

        # Majority Baseline
        maj_model = TrivialMajorityClassifier().fit(X_train, y_train)
        majority_preds.extend(maj_model.predict(X_test))

        # Simple ML Baseline
        simple_model = SimpleMLClassifier().fit(X_train, y_train)
        simple_preds.extend(simple_model.predict(X_test))

        y_true_all.extend(y_test)

    acc_maj = accuracy_score(y_true_all, majority_preds)
    p_maj, r_maj, f1_maj, _ = precision_recall_fscore_support(y_true_all, majority_preds, average='macro', zero_division=0)

    acc_sim = accuracy_score(y_true_all, simple_preds)
    p_sim, r_sim, f1_sim, _ = precision_recall_fscore_support(y_true_all, simple_preds, average='macro', zero_division=0)

    results = {
        "trivial_baseline": {
            "model_name": "Trivial Majority Baseline",
            "accuracy": round(float(acc_maj), 4),
            "precision_macro": round(float(p_maj), 4),
            "recall_macro": round(float(r_maj), 4),
            "f1_macro": round(float(f1_maj), 4)
        },
        "simple_ml_baseline": {
            "model_name": "TF-IDF + Logistic Regression (Simple ML)",
            "accuracy": round(float(acc_sim), 4),
            "precision_macro": round(float(p_sim), 4),
            "recall_macro": round(float(r_sim), 4),
            "f1_macro": round(float(f1_sim), 4)
        }
    }
    return results

if __name__ == "__main__":
    res = evaluate_baselines()
    print("=" * 60)
    print("                BASELINE EVALUATION RESULTS")
    print("=" * 60)
    for model_key, metrics in res.items():
        print(f"Model: {metrics['model_name']}")
        print(f"  - Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  - Precision (Mac): {metrics['precision_macro']:.4f}")
        print(f"  - Recall (Mac):    {metrics['recall_macro']:.4f}")
        print(f"  - F1-Score (Mac):  {metrics['f1_macro']:.4f}")
        print("-" * 60)
