import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_set.csv")
GOLDEN_SET_VERIFIED_PATH = os.path.join(DATA_DIR, "golden_set_verified.csv")

def get_eval_set_path() -> str:
    """Returns human-verified golden set path if present, otherwise candidate golden set."""
    if os.path.exists(GOLDEN_SET_VERIFIED_PATH):
        return GOLDEN_SET_VERIFIED_PATH
    return GOLDEN_SET_PATH

# Evaluation Results
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
RESULTS_PATH = os.path.join(EVAL_DIR, "results.json")

# Selected Brand
SELECTED_BRAND = "AppleSupport"

# Reproducibility
RANDOM_SEED = 42

# Intent Taxonomy Definition for AppleSupport (Clarified Boundaries)
INTENT_TAXONOMY = {
    "hardware_battery": {
        "definition": "Physical hardware issues: drop/impact screen damage, battery health drain, physical charging port damage, or overheating.",
        "examples": [
            "My iPhone screen cracked after dropping it on concrete.",
            "Battery drains from 100% to 10% in two hours.",
            "My phone gets extremely hot to touch while charging."
        ],
        "why_useful": "Routes directly to Genius Bar physical hardware repair & battery diagnostic triage."
    },
    "software_update": {
        "definition": "Operating system update glitches: black screen or boot loop following an update, stuck on Apple logo, post-update WiFi/Bluetooth crashes.",
        "examples": [
            "My iPhone screen went completely black after the latest update and won't turn on!",
            "WiFi keeps disconnecting ever since updating to iOS 17.",
            "My phone is stuck on the Apple logo during software update."
        ],
        "why_useful": "Allows rapid distribution of standard software recovery protocols (forced reboot, iTunes/Finder Recovery Mode)."
    },
    "account_icloud": {
        "definition": "Apple ID lockouts, password resets, two-factor authentication, or iCloud storage limits.",
        "examples": [
            "Forgot my Apple ID password and security questions.",
            "My iCloud storage says full even after deleting photos.",
            "Can't receive 2FA verification codes."
        ],
        "why_useful": "Requires strict verification and secure self-service portal links."
    },
    "billing_refund": {
        "definition": "Double charges, unwanted auto-renewals, App Store purchase disputes, or refund requests.",
        "examples": [
            "I was double charged for my Apple Music subscription this month!",
            "How do I cancel an app subscription that renewed today?",
            "I want a refund for an accidental purchase on the App Store."
        ],
        "why_useful": "Directly impacts customer satisfaction and revenue dispute resolution."
    },
    "accessory_connectivity": {
        "definition": "Pairing, charging, or audio issues with peripheral accessories (AirPods, Apple Watch, Apple Pencil).",
        "examples": [
            "Left AirPod Pro won't connect or charge in the case.",
            "My Apple Watch won't pair with my new phone.",
            "Apple Pencil drops Bluetooth connection constantly."
        ],
        "why_useful": "Separates accessory troubleshooting from core iPhone hardware/software problems."
    },
    "repair_store_genius": {
        "definition": "Scheduling Genius Bar appointments, checking repair progress, or finding Apple Authorized Providers.",
        "examples": [
            "Can I book a Genius Bar appointment for screen replacement online?",
            "How can I check the status of my mail-in repair?",
            "Where is the nearest Apple Store for in-person support?"
        ],
        "why_useful": "Drives foot traffic and logistics to authorized service channels."
    },
    "general_inquiry": {
        "definition": "General feature inquiries, product specs, compatibility checks, or trade-in questions.",
        "examples": [
            "Is the new iPad compatible with Apple Pencil 2nd gen?",
            "How does Apple Trade-In work for older models?",
            "What are the store hours for Apple Store Fifth Avenue?"
        ],
        "why_useful": "Handles high-volume informational queries automatically."
    }
}

# Decision & Escalation Engine Settings
CONFIDENCE_THRESHOLD = 0.65
RETRIEVAL_SIMILARITY_THRESHOLD = 0.25

# Sensitive intents that default to escalation if evidence is weak or complex
SENSITIVE_INTENTS = ["billing_refund", "account_icloud", "repair_store_genius"]
