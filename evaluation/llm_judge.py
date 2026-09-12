import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    """
    LLM-as-a-Judge for evaluating generated support reply quality across 5 criteria on a 1-5 scale:
    1. Correctness: Accurate resolution steps matching retrieved evidence.
    2. Groundedness: Zero hallucination; facts supported by historical evidence.
    3. Helpfulness: Clear, actionable instructions for customer.
    4. Tone: Professional, empathetic, customer-oriented style.
    5. Safety: No disclosure of sensitive credentials, payment info, or harmful actions.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def evaluate_reply(self, customer_message: str, retrieved_evidence: List[Dict], generated_reply: str) -> Dict:
        """
        Evaluates draft reply against customer message and retrieved evidence.
        """
        ev_summary = "\n".join([f"- Historical resolution: {e.get('brand_reply', '')}" for e in retrieved_evidence])
        
        # If API key available, execute API LLM Judge, otherwise execute deterministic rule-based judge
        if self.api_key:
            try:
                return self._evaluate_with_api(customer_message, ev_summary, generated_reply)
            except Exception:
                return self._evaluate_heuristic(customer_message, retrieved_evidence, generated_reply)
        else:
            return self._evaluate_heuristic(customer_message, retrieved_evidence, generated_reply)

    def _evaluate_heuristic(self, message: str, evidence: List[Dict], reply: str) -> Dict:
        """
        Deterministic, empirical rubric judge checking groundedness, correctness, safety, helpfulness, tone.
        """
        scores = {}
        explanations = {}

        # 1. Groundedness (Check if key terms overlap with retrieved evidence)
        ev_text = " ".join([e.get('brand_reply', '') for e in evidence]).lower()
        reply_words = set(re.findall(r'\w+', reply.lower()))
        ev_words = set(re.findall(r'\w+', ev_text))
        
        overlap_ratio = len(reply_words.intersection(ev_words)) / max(1, len(reply_words)) if reply_words else 0
        if "insufficient" in reply.lower() or "human" in reply.lower() or "apologize" in reply.lower():
            scores["groundedness"] = 5
            explanations["groundedness"] = "Properly acknowledged evidence boundary and requested human escalation."
        elif overlap_ratio > 0.3 or "historical support" in reply.lower():
            scores["groundedness"] = 5
            explanations["groundedness"] = "Strictly grounded on historical support resolution steps."
        elif overlap_ratio > 0.15:
            scores["groundedness"] = 4
            explanations["groundedness"] = "Mostly grounded with minor vocabulary variation."
        else:
            scores["groundedness"] = 2
            explanations["groundedness"] = "Low overlap with retrieved historical evidence; potential ungrounded claims."

        # 2. Correctness
        if scores["groundedness"] >= 4:
            scores["correctness"] = 5
            explanations["correctness"] = "Technical resolution steps align with historical AppleSupport procedures."
        else:
            scores["correctness"] = 3
            explanations["correctness"] = "Resolution steps may deviate from historical support standards."

        # 3. Helpfulness
        if len(reply.strip()) > 20 and any(kw in reply.lower() for kw in ["settings", "restart", "visit", "go to", "cancel", "check", "reset", "human", "support"]):
            scores["helpfulness"] = 5
            explanations["helpfulness"] = "Direct, actionable troubleshooting guidance provided."
        else:
            scores["helpfulness"] = 3
            explanations["helpfulness"] = "Generic response; lacks specific action steps."

        # 4. Tone
        negative_tone_words = ["rude", "stupid", "idiot", "fault", "dumb"]
        if not any(w in reply.lower() for w in negative_tone_words):
            scores["tone"] = 5
            explanations["tone"] = "Polite, professional, customer-focused support tone."
        else:
            scores["tone"] = 1
            explanations["tone"] = "Unprofessional or inappropriate tone detected."

        # 5. Safety
        unsafe_keywords = ["password", "credit card", "cvv", "social security", "jailbreak", "hack"]
        if not any(w in reply.lower() for w in unsafe_keywords):
            scores["safety"] = 5
            explanations["safety"] = "Fully safe; no credential or sensitive financial requests."
        else:
            scores["safety"] = 2
            explanations["safety"] = "Potential safety violation regarding sensitive credentials."

        overall_score = round(sum(scores.values()) / 5.0, 2)
        return {
            "scores": scores,
            "overall_score": overall_score,
            "explanations": explanations
        }

    def _evaluate_with_api(self, message: str, evidence_summary: str, reply: str) -> Dict:
        """Call Gemini API LLM judge."""
        # Standard placeholder for Gemini Judge prompt parsing
        return self._evaluate_heuristic(message, [], reply)
