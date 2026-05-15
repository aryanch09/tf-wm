# Raw Data

Raw data should preserve the original episode semantics:

- one JSON, HDF5, Parquet, or simulator log bundle per episode or shard;
- no train-time normalization baked into the source copy;
- metadata that records task, simulator or hardware source, seeds, success,
  camera queries, slip events, and force violations.

Use `scripts/generate_synthetic_dataset.py` for a local smoke-test dataset.
