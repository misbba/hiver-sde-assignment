import unittest
import os
import pandas as pd
from src.retrieval import HistoricalSupportRetriever
from src.preprocessing import clean_text
from src.config import GOLDEN_SET_PATH

class TestRetrievalLeakage(unittest.TestCase):
    """
    Unit tests ensuring retrieval engine prevents data leakage:
    1. A query message cannot retrieve itself (exact or normalized text match).
    2. Excluded tweet IDs are never returned in search results.
    3. Golden evaluation set items are excluded from retrieval when loaded in evaluation mode.
    4. Similarity scores of retrieved results are strictly < 1.0000 when query is excluded.
    """
    def setUp(self):
        self.retriever = HistoricalSupportRetriever()
        self.retriever.load_and_index_data(exclude_golden=True)

    def test_query_cannot_retrieve_itself(self):
        query_msg = "@AppleSupport My iPhone screen went completely black after the latest update and won't turn on!"
        
        # Search using retrieve_similar_examples with explicit exclusion
        results = self.retriever.retrieve_similar_examples(
            query=query_msg,
            exclude_texts=[query_msg],
            top_k=3
        )
        
        # Assert that no retrieved item has normalized text equal to query_msg
        norm_query = clean_text(query_msg)
        for item in results:
            item_norm = clean_text(item['customer_text'])
            self.assertNotEqual(
                item_norm,
                norm_query,
                f"Data leakage detected! Query message '{item['customer_text']}' retrieved itself."
            )
            self.assertLess(
                item['similarity_score'],
                1.0000,
                f"Similarity score is 1.0000 indicating exact duplicate retrieval: {item}"
            )

    def test_tweet_id_exclusion(self):
        query_msg = "WiFi disconnects continuously after iOS 17 update."
        results = self.retriever.retrieve_similar_examples(
            query=query_msg,
            exclude_tweet_ids=["kb_03"],
            top_k=5
        )
        for item in results:
            self.assertNotEqual(
                str(item['id']),
                "kb_03",
                "Data leakage detected! Excluded tweet_id 'kb_03' was returned in retrieval results."
            )

    def test_golden_set_exclusion(self):
        if os.path.exists(GOLDEN_SET_PATH):
            df_golden = pd.read_csv(GOLDEN_SET_PATH)
            golden_sample = df_golden.iloc[0]['message']
            
            results = self.retriever.retrieve_similar_examples(
                query=golden_sample,
                exclude_texts=[golden_sample],
                top_k=3
            )
            norm_sample = clean_text(golden_sample)
            for item in results:
                self.assertNotEqual(
                    clean_text(item['customer_text']),
                    norm_sample,
                    "Golden set item was retrieved as historical evidence!"
                )

if __name__ == "__main__":
    unittest.main()
