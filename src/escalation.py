from typing import Dict, List
from src.config import CONFIDENCE_THRESHOLD, RETRIEVAL_SIMILARITY_THRESHOLD, SENSITIVE_INTENTS

class EscalationEngine:
    """
    Rule-based and threshold-driven conservative escalation decision engine.
    Determines whether a message should be AUTO-HANDLED or ESCALATED TO HUMAN
    with an explicit, human-understandable reason.
    """
    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
        sensitive_intents: List[str] = SENSITIVE_INTENTS
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.sensitive_intents = sensitive_intents

    def evaluate(
        self,
        message: str,
        predicted_intent: str,
        confidence: float,
        top_similarity: float,
        reply_meta: Dict,
        confidence_threshold: float = None
    ) -> Dict[str, str]:
        """
        Evaluates support interaction and returns:
        - action: 'AUTO-HANDLE' or 'ESCALATE'
        - reason: Explicit, human-readable justification
        """
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        msg_lower = message.lower()
        
        # Rule 1: Explicit customer human/agent escalation request keywords
        human_keywords = ["human", "representative", "agent", "supervisor", "lawyer", "sue", "manager"]
        if any(kw in msg_lower for kw in human_keywords):
            return {
                "action": "ESCALATE",
                "reason": "Customer explicitly requested human agent assistance or supervisor intervention."
            }

        # Rule 2: Sensitive intent requiring human verification (billing, accounts, disputes)
        if predicted_intent in self.sensitive_intents:
            return {
                "action": "ESCALATE",
                "reason": f"Predicted intent '{predicted_intent}' involves sensitive financial, security, or repair account operations requiring human verification."
            }

        # Rule 3: Low intent classification confidence
        if confidence < conf_thresh:
            return {
                "action": "ESCALATE",
                "reason": f"Low intent classification confidence ({confidence:.2f} < threshold {conf_thresh:.2f})."
            }

        # Rule 4: Low retrieval similarity / insufficient historical evidence
        if top_similarity < self.similarity_threshold:
            return {
                "action": "ESCALATE",
                "reason": f"Insufficient historical retrieval evidence (Cosine similarity {top_similarity:.2f} < threshold {self.similarity_threshold:.2f})."
            }

        # Rule 5: Ungrounded or fallback reply generator output
        if not reply_meta.get("grounded", True):
            return {
                "action": "ESCALATE",
                "reason": "Reply generator flagged output as ungrounded or lacking verified historical resolution steps."
            }

        # Default: Safe for automated handling
        return {
            "action": "AUTO-HANDLE",
            "reason": f"High confidence intent classification ({confidence:.2f}) and strong historical evidence match ({top_similarity:.2f})."
        }
