# build_data.py - One-time data preparation script.
# Run from project root:  python -m src.build_data
#
# Reads the raw Adult Income CSV, applies cleaning, stratified splitting,
# and preprocessing (QuantileTransformer for numericals, LabelEncoder for
# categoricals), then saves train/val/test splits and metadata as pickle
# files under data/processed/. Only needs to be run once; all downstream
# training and evaluation scripts load the pre-built pickles.

if __name__ == "__main__":
    from src.data_utils import build_dataset

    meta = build_dataset(
        raw_path="data/raw/adult.csv",
        out_dir="data/processed",
        seed=42,   # fixed seed for reproducible splits across experiments
    )



