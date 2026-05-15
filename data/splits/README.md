# Data Splits

Split files list episode paths relative to the repository root. The default
files are:

- `train.txt`
- `val.txt`
- `test.txt`

Generate them from a manifest with:

```bash
python scripts/make_splits.py --manifest data/raw/synthetic/manifest.csv
```
