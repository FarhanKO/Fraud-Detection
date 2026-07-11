# Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the
Machine Learning Group at Université Libre de Bruxelles (ULB), in collaboration
with Worldline.

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Permanent mirror (DOI, no login required):** https://doi.org/10.5281/zenodo.7395559
- **Size:** 284,807 transactions, 31 columns, ~150 MB (not committed to this repo — see below)
- **Timeframe:** Transactions made by European cardholders over two days in September 2013
- **Class balance:** 492 fraudulent transactions out of 284,807 (0.172%) — highly imbalanced

## Columns

| Column | Description |
|---|---|
| `Time` | Seconds elapsed between this transaction and the first transaction in the dataset |
| `V1`–`V28` | Principal components from a PCA transformation, released this way for confidentiality — original features are not disclosed |
| `Amount` | Transaction amount |
| `Class` | Target: `1` = fraud, `0` = legitimate |

## Why the raw file isn't in this repo

The CSV is excluded via `.gitignore` because of its size and because Kaggle's
license does not permit redistribution. To reproduce this project:

1. Download `creditcard.csv` from the Kaggle link above (requires a free Kaggle account), or via the CLI:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
   ```
2. Place it at `data/creditcard.csv`.
3. `src/preprocessing.py::load_data()` expects this path by default.

## License

The dataset is released under the **Open Database License (ODbL) v1.0**. See
the Kaggle page for full terms. This repo contains no copies of the data
itself — only code that operates on it.
