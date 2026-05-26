# Cardinality-Adjusted Noise Schedules for Tabular Diffusion

Per-feature noise schedules for [TabDDPM](https://arxiv.org/abs/2209.15421) derived from mutual-information equalisation, improving synthesis quality for high-cardinality categorical features.


---

## Motivation

TabDDPM (Kotelnikov et al., 2023) extends DDPMs to mixed-type tabular data by combining Gaussian diffusion for numerical features with multinomial diffusion for categoricals. However, it applies a **single noise schedule to every categorical feature regardless of cardinality K**.

Information theory reveals a mismatch: at any shared noise level, mutual information between clean and noisy values decays at different rates across columns. At the midpoint of the schedule (ᾱ_t = 0.5), a binary feature retains 0.13 nats while a 41-category feature retains 1.23 nats - a 9.4x ratio. The model therefore receives inconsistent training signal across features.

This project derives a per-feature noise schedule by solving an analytical mutual-information equalisation condition for each cardinality. The fix is **zero-parameter, precomputed once before training, and architecturally invisible** to the rest of TabDDPM - only the schedule changes.

---

## Headline results (UCI Adult, 3 seeds)

| Method | Mean JSD | Corr L2 | XGBoost F1 |
|---|---|---|---|
| Baseline (TabDDPM) | 0.038 ± 0.000 | 0.556 ± 0.009 | 0.652 ± 0.010 |
| **MI-Adjusted (ours)** | **0.015 ± 0.001** | **0.421 ± 0.002** | **0.660 ± 0.010** |
| Naive β-scaling | 0.093 ± 0.001 | 0.688 ± 0.013 | 0.664 ± 0.007 |

- **61% mean JSD reduction** versus the baseline
- **86% improvement** on native-country (K=41, the highest-cardinality column)
- Naive β-scaling in the opposite direction degrades performance, confirming the information-theoretic derivation is necessary - not just any per-feature adjustment

Per-feature and per-cardinality-bin breakdowns are in `outputs/figures/fig_jsd_per_feature.png` and `outputs/figures/fig_jsd_by_cardinality.png`.

---

## Quick start

### Requirements

- Python 3.10
- PyTorch 2.1 (CUDA recommended)
- See `requirements.txt` for full pin list

### Setup

```bash
git clone https://github.com/RavisaraHasaranga/cardinality-adjusted-diffusion.git
cd cardinality-adjusted-diffusion
pip install -r requirements.txt
```

### Get the data

The UCI Adult dataset is not committed (gitignored). Download `adult.csv` (with header row) and place it at `data/raw/adult.csv`. Two convenient sources:

```python
# Option 1: via OpenML (recommended, header included)
from sklearn.datasets import fetch_openml
import os
os.makedirs("data/raw", exist_ok=True)
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame
df.to_csv("data/raw/adult.csv", index=False)
```

```bash
# Option 2: direct from UCI (no header - you must add one)
mkdir -p data/raw
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
  -o data/raw/adult.csv
# then prepend a header line with: age,workclass,fnlwgt,education,education-num,marital-status,
# occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
```

The expected columns are listed in `src/data_utils.py` (`NUM_COLS`, `CAT_COLS`, `TARGET_COL`).

### Build splits

```bash
python -m src.build_data
```

This cleans the raw CSV, drops rows with missing values, performs a stratified 80/10/10 split with `random_state=42`, fits a `QuantileTransformer` on numericals (train-only) and `LabelEncoder` on categoricals, and writes pickled splits and metadata to `data/processed/`. The script is idempotent and takes a few seconds.

### Train and evaluate

Each notebook is self-contained and reproduces a section of the report:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset inspection, cardinality spread, class balance |
| `02_schedule_visualisation.ipynb` | Plot baseline vs MI-equalised schedules and MI curves |
| `03_baseline_tabddpm.ipynb` | Train TabDDPM with shared schedule (baseline) |
| `04_adjusted_tabddpm.ipynb` | Train TabDDPM with MI-equalised per-feature schedule |
| `05_naive_scaling.ipynb` | Naive β × K_ref/K_i ablation |
| `06_comparison.ipynb` | Aggregate metrics across all methods and seeds |

Run them in order. Each training notebook executes three seeds (42, 1, 7) sequentially - approximately 30 minutes per seed on an RTX 3060, so plan for ~90 minutes per method.

---

## Hardware

Developed and tested on a single **NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)**. The implementation also fits within a free Google Colab T4 session. No multi-GPU or distributed training is required.

For a faster sanity check, reduce `n_train_steps` in the training notebooks from 50,000 to, e.g., 5,000 - the qualitative trends (adjusted < baseline < naive on JSD) remain visible.

---

## Repository structure

```
cardinality-adjusted-diffusion/
├── src/
│   ├── build_data.py              # One-time data preparation (python -m src.build_data)
│   ├── data_utils.py              # Loading, splits, transforms, feature metadata
│   ├── mlp.py                     # Shared MLP backbone (~580K parameters)
│   ├── gaussian_diffusion.py      # Gaussian DDPM for numerical features
│   ├── multinomial_diffusion.py   # Multinomial diffusion (shared and per-feature schedules)
│   ├── schedule_utils.py          # MI calculator + scipy.optimize.brentq solver
│   ├── sampling.py                # Joint reverse sampling loop
│   ├── eval_metrics.py            # JSD, correlation L2, TSTR
│   └── train.py                   # Hybrid training loop (AdamW, cosine LR)
├── notebooks/                     # 01–06, see table above
├── outputs/
│   ├── figures/                   # All report figures (PNG)
│   └── metrics/                   # Per-method, per-seed JSON
├── data/
│   ├── raw/                       # adult.csv (gitignored - see Setup)
│   └── processed/                 # Built splits (gitignored)
├── requirements.txt
├── LICENSE                        # MIT
└── README.md
```

---

## Method in one paragraph

For a categorical feature with K categories under uniform-noise multinomial diffusion, the mutual information between the clean token x₀ and the noisy token x_t is

```
I(x₀; x_t) = log K + p_K · log p_K + (K − 1) · q_K · log q_K
```

where p_K = ᾱ_t + (1 − ᾱ_t)/K and q_K = (1 − ᾱ_t)/K. I is monotonic in ᾱ_t, so for any target MI value there is a unique ᾱ_t. The equalisation condition sets

```
I(ᾱ_t^(K_i), K_i) = I(ᾱ_t_base, K_ref)
```

for every feature i and every timestep t, with K_ref = 2 (binary). This is solved with `scipy.optimize.brentq` at each of T = 1000 timesteps. The resulting per-feature ᾱ_t schedule is converted back to per-step β_t values and passed to `multinomial_diffusion.py`. Every other component of TabDDPM is unchanged.

Implementation: `src/schedule_utils.py` (solver) and `src/multinomial_diffusion.py` (consumption).

---

## Reproducibility

- Seeds 42, 1, 7 are fixed in each notebook
- Schedule precomputation is deterministic
- Data splits are deterministic given `random_state=42` in `build_data.py`
- Per-seed metrics are committed under `outputs/metrics/` as JSON; figures are committed under `outputs/figures/`
- Trained checkpoints are gitignored due to size; rerun the relevant notebook to regenerate

---

## Citation

This is a coursework project; if you build on it, please cite the underlying papers:

- Ho, J., Jain, A. and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P. and Welling, M. (2021). *Argmax Flows and Multinomial Diffusion*. NeurIPS.
- Kotelnikov, A., Baranchuk, D., Rubachev, I. and Babenko, A. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML.

---

## License

MIT - see `LICENSE`.
