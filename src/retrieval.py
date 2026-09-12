import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text
from src.data_loader import detect_dataset_files, inspect_dataset, get_raw_data_dir
from src.config import GOLDEN_SET_PATH, SELECTED_BRAND

class HistoricalSupportRetriever:
    """
    TF-IDF Vector Space Retriever matching incoming customer messages against
    historical AppleSupport conversation pairs (customer_tweet, brand_reply).
    Includes strict filtering and deduplication to prevent evaluation data leakage.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(preprocessor=clean_text, stop_words='english', min_df=1)
        self.knowledge_base: List[Dict] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def build_index_from_pairs(self, pairs: List[Dict]):
        """
        Index historical (customer_tweet, brand_reply) pairs after deduplicating customer text.
        Each pair dict: {'customer_text': str, 'brand_reply': str, 'intent': str, 'id': str}
        """
        if not pairs:
            raise ValueError("No conversation pairs provided to build retrieval index.")
            
        # Deduplicate knowledge base based on normalized customer text
        seen_texts: Set[str] = set()
        deduped_pairs = []
        for pair in pairs:
            norm_text = clean_text(pair['customer_text'])
            if norm_text and norm_text not in seen_texts:
                seen_texts.add(norm_text)
                deduped_pairs.append(pair)
                
        self.knowledge_base = deduped_pairs
        texts = [p['customer_text'] for p in deduped_pairs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_indexed = True
        return self

    def load_and_index_data(self, exclude_golden: bool = True):
        """Build index from data/raw dataset, excluding golden evaluation set items."""
        files = detect_dataset_files()
        if not files:
            pairs = self._get_fallback_support_pairs()
        else:
            file_path = files[0]
            df = pd.read_csv(file_path)
            pairs = self._extract_brand_pairs(df)
            
        # Exclude golden set text if requested to avoid evaluation data leakage
        if exclude_golden and os.path.exists(GOLDEN_SET_PATH):
            golden_df = pd.read_csv(GOLDEN_SET_PATH)
            golden_texts = set(golden_df['message'].apply(clean_text))
            pairs = [p for p in pairs if clean_text(p['customer_text']) not in golden_texts]

        if not pairs:
            pairs = self._get_fallback_support_pairs()
            
        self.build_index_from_pairs(pairs)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Backward-compatible search wrapper enforcing query exclusion by default."""
        return self.retrieve_similar_examples(query=query, exclude_texts=[query], top_k=top_k)

    def retrieve_similar_examples(
        self,
        query: str,
        exclude_tweet_ids: Optional[List[str]] = None,
        exclude_texts: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieves top-K similar historical support examples while strictly excluding:
        1. Current query text (normalized match)
        2. Specified tweet IDs in exclude_tweet_ids
        3. Specified text strings in exclude_texts
        """
        if not self.is_indexed:
            self.load_and_index_data()

        # Build normalized exclusion set
        norm_exclusions: Set[str] = set()
        norm_query = clean_text(query)
        if norm_query:
            norm_exclusions.add(norm_query)
            
        if exclude_texts:
            for txt in exclude_texts:
                cleaned = clean_text(txt)
                if cleaned:
                    norm_exclusions.add(cleaned)

        exclude_ids_set = set(str(tid) for tid in (exclude_tweet_ids or []))

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Sort candidate indices by similarity descending
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            item = self.knowledge_base[idx]
            item_id = str(item.get('id', ''))
            item_norm_text = clean_text(item.get('customer_text', ''))
            
            # Check ID leakage
            if item_id and item_id in exclude_ids_set:
                continue
                
            # Check Text leakage (exact or normalized duplicate)
            if item_norm_text in norm_exclusions:
                continue

            score = float(similarities[idx])
            result_item = dict(item)
            result_item['similarity_score'] = round(score, 4)
            results.append(result_item)
            
            if len(results) >= top_k:
                break
                
        return results

    def _extract_brand_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """Pairs customer inbound tweets with outbound AppleSupport replies."""
        pairs = []
        if 'inbound' not in df.columns or 'author_id' not in df.columns:
            return self._get_fallback_support_pairs()

        brand_responses = df[(df['inbound'] == False) & (df['author_id'].str.lower() == SELECTED_BRAND.lower())]
        
        for _, resp_row in brand_responses.iterrows():
            parent_id = resp_row.get('in_response_to_tweet_id')
            if pd.notna(parent_id):
                cust_rows = df[df['tweet_id'] == parent_id]
                if not cust_rows.empty:
                    cust_text = cust_rows.iloc[0]['text']
                    brand_text = resp_row['text']
                    pairs.append({
                        "id": str(resp_row['tweet_id']),
                        "customer_text": cust_text,
                        "brand_reply": brand_text,
                        "brand": SELECTED_BRAND
                    })
        return pairs if pairs else self._get_fallback_support_pairs()

    def _get_fallback_support_pairs(self) -> List[Dict]:
        """Structured seed pairs for AppleSupport RAG grounding."""
        return [
            {"id": "kb_01", "customer_text": "iPhone screen is black and phone won't turn on.", "brand_reply": "@customer We're here to help! Try a forced restart by pressing Volume Up, Volume Down, then hold Side button until Apple logo appears.", "brand": "AppleSupport"},
            {"id": "kb_02", "customer_text": "Battery drains fast from 100 to 10 in two hours.", "brand_reply": "@customer Battery health is vital. Please check Settings > Battery > Battery Health and send us a DM with your maximum capacity percentage.", "brand": "AppleSupport"},
            {"id": "kb_03", "customer_text": "WiFi disconnects continuously after iOS 17 update.", "brand_reply": "@customer Let's fix that! Try resetting network settings via Settings > General > Transfer or Reset iPhone > Reset Network Settings.", "brand": "AppleSupport"},
            {"id": "kb_04", "customer_text": "Phone stuck on Apple logo during system software update.", "brand_reply": "@customer Connect your iPhone to iTunes/Finder on a Mac/PC and put it into Recovery Mode to complete update/restore.", "brand": "AppleSupport"},
            {"id": "kb_05", "customer_text": "Forgot Apple ID password and locked out of account.", "brand_reply": "@customer You can reset your password securely by visiting iforgot.apple.com or using the Apple Support app on a trusted device.", "brand": "AppleSupport"},
            {"id": "kb_06", "customer_text": "iCloud storage full notification even after deleting photos.", "brand_reply": "@customer Deleted photos stay in 'Recently Deleted' for 30 days. Be sure to empty that folder in the Photos app to free space.", "brand": "AppleSupport"},
            {"id": "kb_07", "customer_text": "Double charged for Apple Music subscription this month.", "brand_reply": "@customer We can help review charges. Visit reportaproblem.apple.com to check purchase history and request a refund.", "brand": "AppleSupport"},
            {"id": "kb_08", "customer_text": "How do I cancel active app subscription renewal on iPhone?", "brand_reply": "@customer Go to Settings > [Your Name] > Subscriptions, tap the subscription, and select Cancel Subscription.", "brand": "AppleSupport"},
            {"id": "kb_09", "customer_text": "Left AirPod Pro won't charge or connect in case.", "brand_reply": "@customer Clean charging contacts inside the case and AirPod stem, then hold the setup button on the case to reset.", "brand": "AppleSupport"},
            {"id": "kb_10", "customer_text": "Can I book Genius Bar screen repair appointment online?", "brand_reply": "@customer Yes! You can schedule an appointment via support.apple.com or the official Apple Support iOS app.", "brand": "AppleSupport"}
        ]
