import re

def clean_text(text: str) -> str:
    """
    Cleans raw customer tweet text by stripping URLs, Twitter handles, extra whitespace,
    and normalizing casing for intent classification and RAG similarity retrieval.
    """
    if not isinstance(text, str):
        return ""
        
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Twitter user handles (e.g., @AppleSupport, @user_123)
    text = re.sub(r'@\w+', '', text)
    
    # Remove non-ascii or special punctuation noise while keeping essential symbols
    text = re.sub(r'[^\w\s\?\!\%]', ' ', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def extract_keywords(text: str) -> list:
    """Extract simple clean tokens for basic text matching."""
    cleaned = clean_text(text)
    return [word for word in cleaned.split() if len(word) > 2]
