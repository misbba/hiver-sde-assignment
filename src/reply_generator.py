import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class ReplyGenerator:
    """
    LLM Reply Generator grounded strictly on retrieved historical support evidence.
    Supports API calls (via GEMINI_API_KEY or OPENAI_API_KEY) and includes a 
    deterministic offline fallback mode for instant demonstration without API keys.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.use_api = bool(self.api_key)

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict],
        top_similarity: float
    ) -> Dict:
        """
        Drafts a reply grounded on retrieved evidence using LLM or deterministic fallback.
        """
        # Formulate grounded prompt context
        evidence_text = ""
        for i, ev in enumerate(retrieved_evidence, 1):
            evidence_text += f"Example {i} [Similarity: {ev.get('similarity_score', 0)}]:\n"
            evidence_text += f"  Customer query: {ev.get('customer_text', '')}\n"
            evidence_text += f"  Historical resolution: {ev.get('brand_reply', '')}\n\n"

        prompt = self._build_prompt(customer_message, predicted_intent, evidence_text)

        # Check if similarity is too low for grounded generation
        if top_similarity < 0.15 or not retrieved_evidence:
            return {
                "draft_reply": "We apologize, but we cannot find verified resolution steps for your specific query. A human customer support agent will assist you shortly.",
                "grounded": False,
                "mode": "fallback_insufficient_evidence",
                "prompt_used": prompt
            }

        if self.use_api:
            # Call API if key available (e.g. Gemini / OpenAI SDK)
            try:
                reply = self._call_llm_api(prompt)
                return {
                    "draft_reply": reply,
                    "grounded": True,
                    "mode": "llm_api",
                    "prompt_used": prompt
                }
            except Exception as e:
                # Gracefully fallback on API error
                fallback_reply = self._deterministic_grounded_reply(customer_message, predicted_intent, retrieved_evidence)
                return {
                    "draft_reply": fallback_reply,
                    "grounded": True,
                    "mode": f"deterministic_fallback (API error: {str(e)})",
                    "prompt_used": prompt
                }
        else:
            # Deterministic offline mock mode based strictly on top retrieved historical resolution
            reply = self._deterministic_grounded_reply(customer_message, predicted_intent, retrieved_evidence)
            return {
                "draft_reply": reply,
                "grounded": True,
                "mode": "deterministic_mock_offline",
                "prompt_used": prompt
            }

    def _build_prompt(self, message: str, intent: str, evidence: str) -> str:
        return f"""You are a helpful customer support assistant drafting responses for Apple customer service inquiries.

INSTRUCTIONS:
1. Answer ONLY using information supported by the retrieved historical support examples below.
2. Do NOT invent policies, monetary refunds, guarantees, specific timelines, or unsupported actions.
3. Be concise, direct, and professional.
4. If evidence is insufficient, state that human support is required.
5. Preserve historical support tone without falsely claiming to be an authorized employee.

CUSTOMER INQUIRY:
"{message}"

PREDICTED INTENT CATEGORY:
{intent}

RETRIEVED HISTORICAL SUPPORT EVIDENCE:
{evidence}

DRAFT GROUNDED REPLY:"""

    def _call_llm_api(self, prompt: str) -> str:
        """Invokes LLM API if key is present."""
        # Generic API caller placeholder for Gemini/OpenAI
        import urllib.request
        import json
        
        # If Gemini API Key present
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
                
        raise NotImplementedError("No supported active API key configured.")

    def _deterministic_grounded_reply(self, message: str, intent: str, evidence: List[Dict]) -> str:
        """Synthesizes clean response grounded strictly on top retrieved brand resolution."""
        top_ev = evidence[0]
        hist_reply = top_ev.get('brand_reply', '')
        
        # Clean brand handle tags from historical text
        import re
        clean_resp = re.sub(r'@\w+', '', hist_reply).strip()
        
        return f"Based on historical support resolution steps: {clean_resp}"
