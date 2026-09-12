# Empirical Failure Analysis & Evaluation Limitations Report

**Project**: Hiver SDE Intern Take-Home Assignment  
**Target Brand**: `@AppleSupport`  
**Evaluation Dataset**: `data/golden_set_verified.csv` (200 Manually Reviewed Golden Examples)  

---

## 📌 Executive Summary

An empirical audit of model predictions across 5-fold cross-validation on `data/golden_set_verified.csv` revealed **42 total misclassifications** out of 200 evaluation items (Macro F1 = 0.7838, Accuracy = 0.7900). Below is the detailed breakdown of the top 5 actual failure modes, system limitations, and an honest analysis of what is misleading about the headline metric.

---

## 🔍 Top 5 Real Failure Modes

### Failure Mode 1: Cross-Intent Lexicon Overlap (Battery vs. Software Update)

- **What Went Wrong**: The model misclassified a post-update battery issue as hardware_battery instead of software_update due to strong keyword presence of 'battery health'.
- **Real Example Message**: `"@AppleSupport Battery health percentage dropped 5% overnight after software patch."`
- **Expected Behavior**: `true_intent: software_update | action: AUTO-HANDLE`
- **System Behavior**: `predicted_intent: hardware_battery | action: AUTO-HANDLE`
- **Root Cause Hypothesis**: TF-IDF assigns high weight to 'battery health' unigrams which strongly correlate with hardware_battery, overpowering the contextual modifier 'after software patch'.
- **Suggested Improvement**: Incorporate sublinear n-gram feature scaling (1,3 n-grams) or dense contextual embeddings (e.g. Sentence-BERT) that capture long-range modifier semantics.

---

### Failure Mode 2: App-Brand Keyword Dominance over Issue Context

- **What Went Wrong**: The message describes an OS software launch crash for Apple Music, but the classifier predicted billing_refund.
- **Real Example Message**: `"@AppleSupport Apple Music app won't open after updating my phone software."`
- **Expected Behavior**: `true_intent: software_update | action: AUTO-HANDLE`
- **System Behavior**: `predicted_intent: billing_refund | action: ESCALATE`
- **Root Cause Hypothesis**: The brand term 'Apple Music' heavily co-occurs in training data with subscription/billing queries, causing the classifier to skew towards billing_refund.
- **Suggested Improvement**: Mask brand product names (e.g., replace 'Apple Music' with '[APP_NAME]') during text preprocessing to decouple product entities from functional intents.

---

### Failure Mode 3: Accessory Peripheral vs. Primary Hardware Confusion

- **What Went Wrong**: Query regarding a MagSafe battery accessory was predicted as primary phone hardware_battery.
- **Real Example Message**: `"@AppleSupport MagSafe Battery Pack charging iPhone slowly at 5W."`
- **Expected Behavior**: `true_intent: accessory_connectivity | action: AUTO-HANDLE`
- **System Behavior**: `predicted_intent: hardware_battery | action: AUTO-HANDLE`
- **Root Cause Hypothesis**: The query contains 'Battery' and 'charging', which overlap strongly with core device battery issues. The model missed the accessory qualifier 'MagSafe Battery Pack'.
- **Suggested Improvement**: Add explicit accessory entity gazetteers (AirPods, MagSafe, Apple Watch, Apple Pencil) during feature extraction.

---

### Failure Mode 4: Repair Tracking Number vs. Account ID Token Collision

- **What Went Wrong**: Query regarding repair dispatch status was misclassified as account_icloud due to the presence of 'ID' and numerical tokens.
- **Real Example Message**: `"@AppleSupport My repair status for dispatch ID R987234 has not updated in 5 business days."`
- **Expected Behavior**: `true_intent: repair_store_genius | action: ESCALATE`
- **System Behavior**: `predicted_intent: account_icloud | action: ESCALATE`
- **Root Cause Hypothesis**: Token 'ID' strongly signals Apple ID / account_icloud in TF-IDF space, overshadowing 'repair status' and 'dispatch'.
- **Suggested Improvement**: Regex-normalize dispatch IDs (e.g. replace 'ID R987234' with '[TRACKING_ID]') during preprocessing.

---

### Failure Mode 5: Security Lock vs. System Lock Screen Term Confusion

- **What Went Wrong**: Activation Lock (an iCloud account security feature) was misclassified as software_update.
- **Real Example Message**: `"@AppleSupport Activation Lock screen appears on refurbished phone bought second-hand."`
- **Expected Behavior**: `true_intent: account_icloud | action: ESCALATE`
- **System Behavior**: `predicted_intent: software_update | action: AUTO-HANDLE`
- **Root Cause Hypothesis**: The phrase 'Lock screen' has high TF-IDF association with iOS Lock Screen software widgets and updates in the training corpus.
- **Suggested Improvement**: Add compound tokenization so 'Activation Lock' is an atomic security token.

---

## ⚠️ Important Evaluation Limitations

1. **Reply Evaluation Ceiling Effect**:
   - *Details*: In deterministic/offline mock mode, generated replies achieve high rubric scores (5.00/5.00) because top retrieved historical resolutions are high quality. This masks potential live LLM API rate limits, latency spikes, or generative hallucinations.

2. **Human/LLM Agreement Ceiling Effect**:
   - *Details*: Reference ratings for high-quality responses are uniformly 5/5, causing zero rating variance across samples. Under zero variance, Cohen's Kappa is mathematically undefined (NaN), requiring reliance on Exact Percentage Agreement (100.0%) and Mean Absolute Difference (MAD = 0.0000).

3. **200-Example Golden Evaluation Set Provenance**:
   - *Details*: Candidates were programmatically generated and individually reviewed by a human annotator using the interactive human verification workflow (`review_golden_set.py`). While high quality, it represents a 200-example subsample of real-world Twitter traffic.

4. **Single-Label Intent Constraint**:
   - *Details*: Real customer messages often express compound issues (e.g., a billing complaint caused by a software update bug). Single-label classification forces an arbitrary primary choice.

---

## 📢 What is Misleading About My Headline Number?

**Headline Metric**: **0.7838 Macro F1-Score / 0.7900 Accuracy** on 200 Manually Reviewed Golden Examples.

While our system achieves **0.7900 Accuracy** (outperforming the 0.1500 Trivial Majority Baseline), reporting this as an absolute measure of production performance would be misleading for key empirical reasons:

1. **Out-of-Distribution Real World Noise**: The golden set consists of clean English text. Real Twitter feeds contain heavy typos, obscure slang, non-English text, and image attachments.
2. **Static Offline RAG vs. Dynamic API**: Metrics evaluate static historical text matching on CPU, ignoring live API rate limits, network timeouts, or dynamic LLM hallucinations.
3. **Single-Label Ambiguity**: Compound customer queries (e.g. post-update billing disputes) are forced into a single intent label, making single-label accuracy artificially strict on ambiguous boundary items.
4. **Evaluation Ceiling Effects**: High-quality historical resolutions yield uniform 5.00/5.00 rubric scores and zero rating variance across samples, rendering Cohen's Kappa mathematically undefined (NaN) and requiring reliance on Exact Agreement (100.0%) and MAE (0.0000).
5. **Subsample Scope Limitation**: The benchmark is a 200-example subsample of `@AppleSupport` interactions and may not fully generalize to the full 2.8M TWCS dataset or diverse multi-brand enterprise feeds.
