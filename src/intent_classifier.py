import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocessing import clean_text
from src.config import INTENT_TAXONOMY, RANDOM_SEED

class TrivialMajorityClassifier:
    """Predicts the most frequent (majority) intent class from training data."""
    def __init__(self):
        self.majority_class = None
        self.classes_ = None
        
    def fit(self, X: List[str], y: List[str]):
        y_series = pd.Series(y)
        self.majority_class = y_series.mode()[0]
        self.classes_ = sorted(list(set(y)))
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        return [self.majority_class] * len(X)
        
    def predict_proba(self, X: List[str]) -> np.ndarray:
        probas = np.zeros((len(X), len(self.classes_)))
        idx = self.classes_.index(self.majority_class)
        probas[:, idx] = 1.0
        return probas

class SimpleMLClassifier:
    """TF-IDF Vectorizer + Logistic Regression Classifier (Baseline 2)."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(preprocessor=clean_text, stop_words='english')
        self.model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        self.classes_ = None
        
    def fit(self, X: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.classes_ = list(self.model.classes_)
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(X)
        return list(self.model.predict(X_vec))
        
    def predict_proba(self, X: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)

class IntentClassifier:
    """
    Optimized TF-IDF (1,2 n-grams, sublinear TF) + Calibrated Logistic Regression Classifier
    with class balancing and confidence scoring.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            min_df=1
        )
        self.model = LogisticRegression(
            C=1.5,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            max_iter=1000
        )
        self.classes_ = None
        self.is_fitted = False
        
    def fit(self, X: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        X_vec = self.vectorizer.transform(X)
        return list(self.model.predict(X_vec))
        
    def predict_with_confidence(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict intent with top confidence score and probability distribution."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        X_vec = self.vectorizer.transform([text])
        probas = self.model.predict_proba(X_vec)[0]
        top_idx = int(np.argmax(probas))
        predicted_intent = self.classes_[top_idx]
        confidence = float(probas[top_idx])
        
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes_, probas)}
        return predicted_intent, confidence, prob_dict
