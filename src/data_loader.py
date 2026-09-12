import os
import glob
import pandas as pd
from typing import Dict, List, Tuple, Optional

def get_raw_data_dir() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "raw")

def detect_dataset_files(data_dir: Optional[str] = None) -> List[str]:
    """Search for raw CSV or JSON files in data/raw."""
    if data_dir is None:
        data_dir = get_raw_data_dir()
    
    if not os.path.exists(data_dir):
        return []
        
    patterns = ["*.csv", "*.csv.gz", "*.json"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))
    return sorted(files)

def inspect_dataset(file_path: str, nrows: int = 50000) -> Dict:
    """
    Inspect raw dataset automatically without hardcoding assumptions.
    Returns metadata: filename, row_count, columns, sample_records, brand_counts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
    file_name = os.path.basename(file_path)
    
    # Read header and sample
    if file_path.endswith('.json'):
        df_sample = pd.read_json(file_path)
        total_rows = len(df_sample)
    else:
        df_sample = pd.read_csv(file_path, nrows=nrows)
        # Count total rows safely
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_rows = sum(1 for _ in f) - 1  # header excluded
            
    columns = list(df_sample.columns)
    sample_records = df_sample.head(3).to_dict(orient='records')
    
    # Detect brand names based on author_id or inbound flags
    brand_counts = {}
    if 'author_id' in df_sample.columns:
        if 'inbound' in df_sample.columns:
            brand_df = df_sample[df_sample['inbound'] == False]
            brand_counts = brand_df['author_id'].value_counts().to_dict()
        else:
            non_numeric = df_sample[df_sample['author_id'].astype(str).str.contains(r'[a-zA-Z_]')]
            brand_counts = non_numeric['author_id'].value_counts().to_dict()
            
    return {
        "file_name": file_name,
        "file_path": file_path,
        "total_rows": total_rows,
        "columns": columns,
        "sample_records": sample_records,
        "brand_counts": brand_counts
    }
