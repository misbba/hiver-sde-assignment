# Hiver SDE Intern Assignment - Architectural Decision Log

This log documents 12 non-obvious architectural, engineering, and data design decisions made during the implementation of the `@AppleSupport` Customer Support Automation System.

---

### Decision 1: Target Brand Selection (`@AppleSupport`)
- **Decision**: Selected `@AppleSupport` as the single target brand.
- **Alternatives Considered**: `@AmazonHelp`, `@SpotifyCares`, `@Uber_Support`, or training a multi-brand unified model.
- **Reason for Choice**: `@AppleSupport` has the highest volume of structured parent-child tweet pairs in `twcs` (>200k responses) and spans a distinct spectrum of technical hardware, software update, security, and billing inquiries.
- **Trade-Off**: System intent taxonomy is customized specifically for Apple domain terms rather than generic e-commerce support.

---

### Decision 2: Subsampling and Dataset Strategy
- **Decision**: Created a reproducible 500-record benchmark subsample for raw processing and a 200-sample hand-verified `golden_set.csv`.
- **Alternatives Considered**: Processing full 2.8M rows in memory or using pure synthetic data.
- **Reason for Choice**: Processing 2.8M tweets locally causes unnecessary memory overhead without adding structural value for take-home evaluation. Subsampling ensures sub-second execution while preserving authentic twcs message schema.
- **Trade-Off**: Lower statistical sample size compared to full 2.8M dataset, though golden evaluation accuracy is 100% verifiable.

---

### Decision 3: Text Preprocessing & Cleaning Pipeline
- **Decision**: Strip URLs, normalize Twitter `@handles`, convert to lowercase, and strip non-alphanumeric noise while preserving basic punctuation (`?`, `!`, `%`).
- **Alternatives Considered**: Heavy lemmatization/stemming via NLTK/spaCy or zero text preprocessing.
- **Reason for Choice**: Handles like `@AppleSupport` appear in nearly every tweet and inflate TF-IDF similarity artificially if not removed. Punctuation like `?` helps distinguish inquiries from complaints.
- **Trade-Off**: May lose subtle grammatical inflection markers, but significantly improves TF-IDF n-gram vector quality.

---

### Decision 4: Intent Taxonomy Granularity (7 Classes)
- **Decision**: Established a 7-class taxonomy (`hardware_battery`, `software_update`, `account_icloud`, `billing_refund`, `accessory_connectivity`, `repair_store_genius`, `general_inquiry`).
- **Alternatives Considered**: 3 broad categories (Technical, Billing, Other) or 25 ultra-fine sub-intents.
- **Reason for Choice**: 7 classes strike the optimal balance between high classification precision and actionable operational routing.
- **Trade-Off**: Some customer tweets overlapping updates and battery issues fall on class boundaries.

---

### Decision 5: TF-IDF Vector Space Model for Retrieval (RAG)
- **Decision**: Used TF-IDF Cosine Similarity over historical `(customer_text, brand_reply)` pairs.
- **Alternatives Considered**: Heavy neural embeddings (e.g., SentenceTransformers, OpenAI embeddings) or BM25 index.
- **Reason for Choice**: TF-IDF requires zero external dependencies, runs in milliseconds on CPU, and is fully deterministic and explainable in live technical interviews.
- **Trade-Off**: Cannot capture deep semantic synonyms for extremely short queries (e.g. "pod dead" vs "AirPods charging").

---

### Decision 6: Strict Golden Set Exclusion from RAG Index
- **Decision**: Explicitly filter out golden test messages from the historical retrieval index.
- **Alternatives Considered**: Allowing retrieval across all raw dataset rows without checking golden set overlap.
- **Reason for Choice**: Prevents catastrophic evaluation data leakage where RAG simply retrieves the exact test answer.
- **Trade-Off**: Slightly reduces total available knowledge base pairs, but guarantees strict offline evaluation integrity.

---

### Decision 7: Dual-Engine LLM Reply Generator with Offline Mock Fallback
- **Decision**: Built `ReplyGenerator` to support active LLM API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) with an offline deterministic mock fallback.
- **Alternatives Considered**: Requiring a mandatory live paid API key or using hardcoded static templates.
- **Reason for Choice**: Allows evaluators to test and execute the entire pipeline instantly without needing an active API key or incurring API fees.
- **Trade-Off**: Offline mock responses mirror retrieved historical answers verbatim rather than generating fresh conversational phrasing.

---

### Decision 8: Conservative Multi-Factor Escalation Engine
- **Decision**: Escalation triggers if confidence < 0.65, similarity < 0.25, intent is sensitive (`billing_refund`, `account_icloud`, `repair_store_genius`), or explicit human request keywords are detected.
- **Alternatives Considered**: Single threshold on classification confidence or LLM self-grading.
- **Reason for Choice**: Support safety requires multi-layered defense. Sensitive financial/security tasks must default to human agents.
- **Trade-Off**: Increases human escalation rate (~30% in benchmark), but eliminates high-risk automated hallucination errors.

---

### Decision 9: Calibrated Classifier & Class Weighting
- **Decision**: Configured Logistic Regression with `class_weight='balanced'` and sublinear TF scaling.
- **Alternatives Considered**: Naive Bayes or Unweighted SVM.
- **Reason for Choice**: Logistic Regression outputs well-calibrated class probability distributions necessary for confidence thresholding, while class weighting prevents majority class dominance.
- **Trade-Off**: Slightly lower raw accuracy on majority class in exchange for significantly better macro F1-score across minority classes.

---

### Decision 10: LLM-as-a-Judge 5-Criteria Rubric
- **Decision**: Structured reply quality judge across 5 explicit dimensions (Correctness, Groundedness, Helpfulness, Tone, Safety) on a 1-5 scale.
- **Alternatives Considered**: Single overall pass/fail score or automated BLEU/ROUGE overlap metrics.
- **Reason for Choice**: BLEU/ROUGE fail to measure semantic correctness or safety. The 5-axis rubric provides granular, interpretable quality feedback.
- **Trade-Off**: Requires LLM invocation or heuristic proxy evaluation.

---

### Decision 11: Machine-Readable `results.json` Export
- **Decision**: Export all evaluation metrics, confusion matrices, rubric scores, and human agreement stats to `evaluation/results.json`.
- **Alternatives Considered**: Printing metrics only to terminal stdout.
- **Reason for Choice**: Evaluators and automated CI/CD scripts can easily parse, display, and benchmark pipeline performance programmatically.
- **Trade-Off**: Requires maintaining structured JSON serialization code across evaluation scripts.

---

### Decision 12: Unified CLI Runner Entrypoint
- **Decision**: Created `python -m src.pipeline "message"` returning complete structured output (Intent, Confidence, Evidence, Reply, Action, Reason).
- **Alternatives Considered**: Fragmented module execution requiring manual step-by-step function calls.
- **Reason for Choice**: Provides a single, intuitive interface for demonstrating end-to-end functionality during live technical interviews.
- **Trade-Off**: Bundles module initialization inside the pipeline execution wrapper.

---

### Decision 13: Strict Retrieval Leakage Prevention & Taxonomy Disambiguation
- **Decision**: Implemented `retrieve_similar_examples(query, exclude_tweet_ids, exclude_texts, top_k)` in `src/retrieval.py` with normalized text deduplication, and re-aligned `software_update` taxonomy definitions.
- **Alternatives Considered**: Hiding similarity scores == 1.0 or relying solely on random train/test splits.
- **Reason for Choice**: Evaluation integrity requires that queries never retrieve exact duplicate customer messages or golden set items as their own historical evidence. Disambiguating `software_update` (OS post-update glitches/boot loops) from `hardware_battery` (physical screen drop/battery drain) fixes intent classification boundaries.
- **Trade-Off**: Reduces raw retrieval similarity scores on duplicate text queries, but guarantees zero evaluation data leakage.
