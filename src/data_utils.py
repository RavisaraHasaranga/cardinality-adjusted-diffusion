# src/data_utils.py
#
# Data loading and preprocessing pipeline for the UCI Adult Income dataset.
# Produces train/val/test splits with:
#   - QuantileTransformer-normalised numericals (fitted on train only)
#   - LabelEncoder integer-encoded categoricals (fitted on train only)
# All fitted transformers are saved in a 'meta' dict so that generated
# samples can be inverse-transformed back to the original feature space.

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split

# Column definitions --------------------------------
# Adult has 6 continuous and 8 categorical features plus a binary target.

NUM_COLS = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week'
]

CAT_COLS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

TARGET_COL = 'income'


# Loading -----------------------------------------

def load_raw(path: str = "data/raw/adult.csv") -> pd.DataFrame:
    """Load adult.csv, clean whitespace, strip trailing dots, drop missing."""
    # UCI Adult uses '?' for missing values; treat both '?' and ' ?' as NaN
    df = pd.read_csv(path, na_values=['?', ' ?'])

    # Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes('object').columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # The UCI test file appends a trailing dot to income labels (e.g. '<=50K.')
    df[TARGET_COL] = df[TARGET_COL].str.rstrip('.')

    # Drop rows with any missing value (~2,399 rows across workclass/occupation/native-country).
    # This is standard practice for Adult and simplifies cardinality counts.
    df = df.dropna().reset_index(drop=True)

    return df


# Splitting ---------------------------------------------------

def split_data(df: pd.DataFrame, seed: int = 42):
    """80 / 10 / 10 stratified split. Returns (train_df, val_df, test_df)."""
    # Two-stage split: first 80/20, then halve the 20% into val and test.
    # Stratified on the target to preserve class balance in each split.
    train_df, tmp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df[TARGET_COL]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=seed, stratify=tmp_df[TARGET_COL]
    )
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True), \
           test_df.reset_index(drop=True)


# Numerical transform -------------------------------------------------------

def fit_quantile_transformer(train_df: pd.DataFrame, n_quantiles: int = 1000):
    """Fit QuantileTransformer on train numericals only. Returns fitted transformer."""
    # QuantileTransformer maps each numerical feature to a standard normal
    # distribution. This is the same preprocessing used in TabDDPM, making
    # Gaussian diffusion well-suited since the forward process also adds
    # Gaussian noise. Fitted on train only to prevent data leakage.
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        random_state=42
    )
    qt.fit(train_df[NUM_COLS].values)
    return qt


def transform_numericals(df: pd.DataFrame, qt: QuantileTransformer) -> np.ndarray:
    """Apply fitted QuantileTransformer. Returns float32 array (N, 6)."""
    return qt.transform(df[NUM_COLS].values).astype(np.float32)


# Categorical encoding ------------------------------------------------------

def fit_label_encoders(train_df: pd.DataFrame) -> dict:
    """Fit one LabelEncoder per categorical column on train set only.

    Each encoder maps string category names to integer indices [0, K_i).
    These integers are later converted to one-hot vectors by the diffusion module.
    """
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(train_df[col].values)
        encoders[col] = le
    return encoders


def transform_categoricals(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """
    Label-encode each categorical column. Unseen categories in val/test
    are mapped to the closest existing class (via np.searchsorted).
    Returns int64 array (N, 8).
    """
    encoded = np.zeros((len(df), len(CAT_COLS)), dtype=np.int64)
    for i, col in enumerate(CAT_COLS):
        le = encoders[col]
        vals = df[col].values
        # Handle unseen categories gracefully: any value not seen during
        # training is replaced with the first known class to avoid errors.
        known = set(le.classes_)
        safe = np.where(
            pd.Series(vals).isin(known).values,
            vals,
            le.classes_[0]   # fallback to first class for unseen
        )
        encoded[:, i] = le.transform(safe)
    return encoded


def get_cardinalities(encoders: dict) -> list:
    """Return list of K_i (number of categories) for each categorical feature.

    These cardinalities are essential for constructing the multinomial diffusion
    transition matrices and for determining MLP output head dimensions.
    For Adult: [7, 16, 7, 14, 6, 5, 2, 41] (after dropping missing rows).
    """
    return [len(encoders[col].classes_) for col in CAT_COLS]


# Target encoding --------------------------------------------

def encode_target(df: pd.DataFrame) -> np.ndarray:
    """Binary encode income: '>50K' → 1, '<=50K' → 0. Returns int64 array (N,)."""
    return (df[TARGET_COL] == '>50K').astype(np.int64).values


# Full pipeline ---------------------------------------

def build_dataset(
    raw_path: str = "data/raw/adult.csv",
    out_dir:  str = "data/processed",
    seed:     int = 42
) -> dict:
    """
    End-to-end pipeline: load → clean → split → transform → save.

    Saves:
        adult_train.pkl, adult_val.pkl, adult_test.pkl
        adult_meta.pkl   ← fitted transformers + cardinalities + col names

    Each split pkl is a dict:
        {
            'x_num': np.ndarray  (N, 6)  float32  - quantile-transformed numericals
            'x_cat': np.ndarray  (N, 8)  int64    - label-encoded categoricals
            'y':     np.ndarray  (N,)    int64    - binary target
        }

    Returns the meta dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & clean
    df = load_raw(raw_path)
    print(f"[data] Loaded {len(df):,} rows after dropping missing values.")

    # 2. Split
    train_df, val_df, test_df = split_data(df, seed=seed)
    print(f"[data] Split → train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # 3. Fit transforms on TRAIN only (prevents data leakage into val/test)
    qt       = fit_quantile_transformer(train_df)
    encoders = fit_label_encoders(train_df)

    # 4. Apply the train-fitted transforms to all three splits
    splits = {}
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        splits[name] = {
            'x_num': transform_numericals(split_df, qt),   # (N, 6) float32
            'x_cat': transform_categoricals(split_df, encoders),  # (N, 8) int64
            'y':     encode_target(split_df),               # (N,)   int64
        }

    # 5. Cardinalities - used to size MLP output heads and build
    #    multinomial transition matrices in the diffusion modules
    cardinalities = get_cardinalities(encoders)
    print(f"[data] Cardinalities per categorical feature:")
    for col, k in zip(CAT_COLS, cardinalities):
        print(f"         {col:<20s}  K={k}")

    # 6. Meta dict - bundles everything downstream code needs:
    #    - cardinalities for model construction and schedule computation
    #    - fitted transformers for inverse-transforming generated samples
    meta = {
        'num_cols':      NUM_COLS,
        'cat_cols':      CAT_COLS,
        'target_col':    TARGET_COL,
        'cardinalities': cardinalities,       # [K_1, ..., K_8] per categorical feature
        'n_num':         len(NUM_COLS),       # 6
        'n_cat':         len(CAT_COLS),       # 8
        'qt':            qt,                  # fitted QuantileTransformer for inverse at eval
        'cat_encoders':  encoders,            # fitted LabelEncoders for decoding at eval
        'split_sizes':   {k: len(v['y']) for k, v in splits.items()},
        'seed':          seed,
    }

    # 7. Save
    for name, split in splits.items():
        path = out_dir / f"adult_{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(split, f)
        print(f"[data] Saved {path}")

    meta_path = out_dir / "adult_meta.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"[data] Saved {meta_path}")

    return meta


# Loaders (used by training scripts) ---------------------------------------

def load_split(name: str, data_dir: str = "data/processed") -> dict:
    """Load a pre-built split. name ∈ {'train', 'val', 'test'}."""
    path = Path(data_dir) / f"adult_{name}.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_meta(data_dir: str = "data/processed") -> dict:
    """Load dataset metadata (cardinalities, transformers, col names)."""
    path = Path(data_dir) / "adult_meta.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


# Inverse transform (for decoding generated samples) ---------------------------------

def decode_sample(x_num: np.ndarray, x_cat: np.ndarray, meta: dict) -> pd.DataFrame:
    """
    Convert raw model output back to a human-readable DataFrame.
    Used after sampling to get real-valued numericals and string categoricals
    for evaluation metrics (JSD, TSTR, etc.).

      x_num: (N, 6) float32  - still in quantile-normal space
      x_cat: (N, 8) int64    - label-encoded integer indices
    """
    # Inverse quantile transform: normal space -> original feature scales
    num_decoded = meta['qt'].inverse_transform(x_num)
    num_df = pd.DataFrame(num_decoded, columns=meta['num_cols'])

    # Decode integer indices back to original category strings
    cat_decoded = {}
    for i, col in enumerate(meta['cat_cols']):
        cat_decoded[col] = meta['cat_encoders'][col].inverse_transform(x_cat[:, i])
    cat_df = pd.DataFrame(cat_decoded)

    return pd.concat([num_df, cat_df], axis=1)

























