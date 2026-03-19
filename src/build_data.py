# build_data.py  — run once from project root
from data_utils import build_dataset

meta = build_dataset(
    raw_path="data/raw/adult.csv",
    out_dir="data/processed",
    seed=42
)
