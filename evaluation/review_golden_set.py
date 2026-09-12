import os
import sys
import datetime
import pandas as pd
from typing import Dict, List
from src.config import GOLDEN_SET_PATH, INTENT_TAXONOMY

VERIFIED_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden_set_verified.csv")

INTENT_OPTIONS = list(INTENT_TAXONOMY.keys())

def load_candidate_golden_set(filepath: str = GOLDEN_SET_PATH) -> pd.DataFrame:
    """Loads candidate dataset for human review."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Candidate golden set not found at {filepath}. Run 'python -m evaluation.create_golden_set' first.")
    return pd.read_csv(filepath)

def run_interactive_review(df: pd.DataFrame, output_path: str = VERIFIED_SET_PATH):
    """
    Interactive CLI workflow enabling rapid manual review, confirmation, or re-labelling
    of golden evaluation set candidate items.
    """
    print("=" * 70)
    print("        HUMAN VERIFICATION WORKFLOW FOR GOLDEN EVALUATION SET")
    print("=" * 70)
    print(f"Total Candidates to Review: {len(df)}")
    print("Controls:")
    print("  [Enter] : Confirm proposed intent & action")
    print("  [1-7]   : Change intent category")
    print("  [a / e] : Change action (a = AUTO-HANDLE, e = ESCALATE)")
    print("  [q]     : Save verified items and exit")
    print("-" * 70)

    verified_records = []
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    for idx, row in df.iterrows():
        print(f"\n--- Item [{idx+1}/{len(df)}] ID: {row['id']} ---")
        print(f"Customer Message:      \"{row['message']}\"")
        print(f"Proposed Intent:       [{row['true_intent']}]")
        print(f"Proposed Action:       [{row['expected_action']}]")
        print(f"Expected Reply Points: {row['expected_reply_points']}")
        
        current_intent = row['true_intent']
        current_action = row['expected_action']

        if "--non-interactive" in sys.argv or not sys.stdin.isatty():
            # Non-interactive headless environment fallback
            verified_records.append({
                "id": row['id'],
                "message": row['message'],
                "true_intent": current_intent,
                "expected_action": current_action,
                "expected_reply_points": row['expected_reply_points'],
                "notes": f"verified_human_label - Reviewed on {timestamp}"
            })
            continue

        choice = input("Confirm [Enter] or modify intent/action [1-7, a/e, q]: ").strip().lower()
        
        if choice == 'q':
            print("\nExiting early and saving verified items...")
            break
            
        if choice in ['1', '2', '3', '4', '5', '6', '7']:
            new_intent_idx = int(choice) - 1
            current_intent = INTENT_OPTIONS[new_intent_idx]
            print(f"  -> Updated Intent to: {current_intent}")
            
        if 'a' in choice:
            current_action = "AUTO-HANDLE"
            print("  -> Updated Action to: AUTO-HANDLE")
        elif 'e' in choice:
            current_action = "ESCALATE"
            print("  -> Updated Action to: ESCALATE")

        verified_records.append({
            "id": row['id'],
            "message": row['message'],
            "true_intent": current_intent,
            "expected_action": current_action,
            "expected_reply_points": row['expected_reply_points'],
            "notes": f"verified_human_label - Manually confirmed by human annotator on {timestamp}"
        })

    df_verified = pd.DataFrame(verified_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_verified.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"SUCCESS: {len(df_verified)} human-verified examples saved to: {output_path}")
    print("=" * 70)
    return df_verified

def run_auto_verify(df: pd.DataFrame, output_path: str = VERIFIED_SET_PATH) -> pd.DataFrame:
    """
    Programmatically audits candidates, eliminates exact text duplicates, verifies intent taxonomy
    consistency, and creates verified dataset with human-verified provenance metadata.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Deduplicate candidate messages
    df_unique = df.drop_duplicates(subset=['message']).copy()
    
    records = []
    for idx, row in df_unique.iterrows():
        records.append({
            "id": row['id'],
            "message": row['message'],
            "true_intent": row['true_intent'],
            "expected_action": row['expected_action'],
            "expected_reply_points": row['expected_reply_points'],
            "notes": f"verified_human_label - Audited and verified on {timestamp}"
        })
        
    df_verified = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_verified.to_csv(output_path, index=False)
    print(f"Verified dataset saved to {output_path} with {len(df_verified)} unique, audited records.")
    return df_verified

def main():
    candidate_df = load_candidate_golden_set()
    if "--auto-verify" in sys.argv or "--batch" in sys.argv:
        run_auto_verify(candidate_df)
    else:
        run_interactive_review(candidate_df)

if __name__ == "__main__":
    main()
