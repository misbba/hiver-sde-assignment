import os
import sys
import pandas as pd
from src.data_loader import detect_dataset_files, inspect_dataset, get_raw_data_dir
from src.sample_generator import generate_sample_dataset

def main():
    raw_dir = get_raw_data_dir()
    print("=" * 60)
    print("      HIVER SDE ASSIGNMENT - DATASET INSPECTION")
    print("=" * 60)
    print(f"Scanning raw data directory: {raw_dir}\n")

    files = detect_dataset_files(raw_dir)

    if not files:
        print("[NOTICE] No dataset file found in data/raw/.")
        print("Creating benchmark dataset sample at data/raw/sample_twcs.csv...")
        sample_path = os.path.join(raw_dir, "sample_twcs.csv")
        generate_sample_dataset(sample_path)
        files = [sample_path]
        print("To run on full Kaggle dataset, place 'twcs.csv' into data/raw/ directory.\n")

    print(f"Detected {len(files)} raw data file(s):")
    for f in files:
        print(f" - {os.path.basename(f)}")
    print("-" * 60)

    # Inspect primary dataset file
    target_file = files[0]
    info = inspect_dataset(target_file)

    print(f"File Name:      {info['file_name']}")
    print(f"Total Rows:     {info['total_rows']:,}")
    print(f"Column Names:   {info['columns']}")
    print("-" * 60)

    print("Sample Records (First 2):")
    for idx, record in enumerate(info['sample_records'][:2], 1):
        print(f"\nRecord {idx}:")
        for k, v in record.items():
            print(f"  {k}: {v}")

    print("\n" + "-" * 60)
    print("Top Available Brand Accounts & Response Counts:")
    brands = info['brand_counts']
    if brands:
        sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)
        for brand, count in sorted_brands[:10]:
            print(f"  - {brand}: {count:,} responses")
    else:
        print("  No brand accounts automatically identified.")

    print("-" * 60)
    print("Conversation Structure Identification:")
    print("  - 'inbound': True = Customer message, False = Support response")
    print("  - 'tweet_id': Unique message identifier")
    print("  - 'author_id': Customer ID or Brand handle (e.g., AppleSupport)")
    print("  - 'in_response_to_tweet_id': Links brand reply to customer tweet")
    print("  - 'response_by_tweet_id': Links customer tweet to brand reply")
    print("=" * 60)

if __name__ == "__main__":
    main()
