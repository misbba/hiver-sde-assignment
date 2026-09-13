import sys
import os
import pandas as pd
from typing import Dict, Optional, List

from src.preprocessing import clean_text
from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalSupportRetriever
from src.reply_generator import ReplyGenerator
from src.escalation import EscalationEngine
from src.config import GOLDEN_SET_PATH

class SupportPipeline:
    """
    Unified end-to-end Customer Support Automation & Escalation Pipeline.
    Pipeline flow: Message -> Preprocessing -> Intent Classification -> RAG Retrieval -> Reply Drafting -> Escalation Engine.
    Strictly prevents data leakage by excluding query text and test IDs from RAG retrieval.
    """
    def __init__(self, golden_path: Optional[str] = GOLDEN_SET_PATH):
        self.classifier = IntentClassifier()
        self.retriever = HistoricalSupportRetriever()
        self.reply_gen = ReplyGenerator()
        self.escalation_engine = EscalationEngine()
        self._is_trained = False

    def train_pipeline(self, golden_path: Optional[str] = GOLDEN_SET_PATH, exclude_golden_from_retrieval: bool = True):
        """Train classifier and build retrieval index on historical training data."""
        if golden_path and os.path.exists(golden_path):
            df_gold = pd.read_csv(golden_path)
            train_messages = df_gold['message'].tolist()
            train_intents = df_gold['true_intent'].tolist()
        else:
            train_messages = [
                "iPhone battery drains fast", "Screen cracked after dropping phone",
                "WiFi disconnecting after iOS 17 update", "Phone stuck on black screen after software update",
                "Forgot Apple ID password locked out", "iCloud storage full notification",
                "Double charged for Apple Music", "Cancel app subscription",
                "AirPod Pro won't charge in case", "Genius Bar appointment booking"
            ]
            train_intents = [
                "hardware_battery", "hardware_battery",
                "software_update", "software_update",
                "account_icloud", "account_icloud",
                "billing_refund", "billing_refund",
                "accessory_connectivity", "repair_store_genius"
            ]

        self.classifier.fit(train_messages, train_intents)
        self.retriever.load_and_index_data(exclude_golden=exclude_golden_from_retrieval)
        self._is_trained = True

    def process_message(
        self,
        message: str,
        tweet_id: Optional[str] = None,
        exclude_texts: Optional[List[str]] = None,
        evaluation_mode: bool = False,
        confidence_threshold: Optional[float] = None
    ) -> Dict:
        """
        Processes a single customer message through the full pipeline with strict leakage prevention.
        """
        if not self._is_trained:
            self.train_pipeline(exclude_golden_from_retrieval=evaluation_mode)

        # Step 1: Preprocessing
        cleaned_msg = clean_text(message)

        # Step 2: Intent Classification
        predicted_intent, confidence, proba_dist = self.classifier.predict_with_confidence(message)

        # Step 3: Retrieval of Historical Examples with Strict Leakage Exclusion
        exclusions = [message]
        if exclude_texts:
            exclusions.extend(exclude_texts)
            
        exclude_ids = [tweet_id] if tweet_id else []

        retrieved_examples = self.retriever.retrieve_similar_examples(
            query=message,
            exclude_tweet_ids=exclude_ids,
            exclude_texts=exclusions,
            top_k=2
        )
        
        top_similarity = retrieved_examples[0]['similarity_score'] if retrieved_examples else 0.0

        # Step 4: Grounded LLM Reply Generation
        reply_meta = self.reply_gen.generate_reply(
            customer_message=message,
            predicted_intent=predicted_intent,
            retrieved_evidence=retrieved_examples,
            top_similarity=top_similarity
        )

        # Step 5: Conservative Escalation Decision
        escalation_result = self.escalation_engine.evaluate(
            message=message,
            predicted_intent=predicted_intent,
            confidence=confidence,
            top_similarity=top_similarity,
            reply_meta=reply_meta,
            confidence_threshold=confidence_threshold
        )

        return {
            "customer_message": message,
            "cleaned_message": cleaned_msg,
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "intent_probabilities": proba_dist,
            "retrieved_examples": retrieved_examples,
            "top_similarity": top_similarity,
            "draft_reply": reply_meta["draft_reply"],
            "reply_mode": reply_meta["mode"],
            "action": escalation_result["action"],
            "reason": escalation_result["reason"]
        }

def print_pipeline_output(result: Dict):
    print("=" * 65)
    print("                SUPPORT PIPELINE OUTPUT")
    print("=" * 65)
    print(f"Customer Message:   {result['customer_message']}")
    print(f"Predicted Intent:   {result['predicted_intent']} (Confidence: {result['confidence']:.2%})")
    print(f"Top Similarity:     {result['top_similarity']:.4f}")
    print("-" * 65)
    print("Retrieved Evidence (Filtered & Deduplicated):")
    if result['retrieved_examples']:
        for idx, ev in enumerate(result['retrieved_examples'], 1):
            print(f"  [{idx}] Sim: {ev['similarity_score']:.4f} | Customer: \"{ev['customer_text']}\"")
            print(f"      Resolution: \"{ev['brand_reply']}\"")
    else:
        print("  [No non-duplicate historical evidence found above similarity threshold]")
    print("-" * 65)
    print(f"Draft Reply:        {result['draft_reply']}")
    print(f"Generation Mode:    {result['reply_mode']}")
    print("-" * 65)
    print(f"ACTION:             [{result['action']}]")
    print(f"REASON:             {result['reason']}")
    print("=" * 65)

def main():
    if len(sys.argv) > 1:
        user_msg = " ".join(sys.argv[1:])
    else:
        user_msg = "@AppleSupport My iPhone screen went completely black after the latest update and won't turn on!"

    pipeline = SupportPipeline()
    pipeline.train_pipeline(exclude_golden_from_retrieval=True)
    result = pipeline.process_message(user_msg, evaluation_mode=True)
    print_pipeline_output(result)

if __name__ == "__main__":
    main()
