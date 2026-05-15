"""TensorFlow Datasets loader."""

from __future__ import annotations

from ..episode import Episode
from ..schema import Episode as EpisodeSchema


def load_episodes_from_tfds(dataset_name: str, split: str = "train") -> list[Episode]:
    """Load episodes from TensorFlow Datasets.

    Args:
        dataset_name: TFDS dataset name.
        split: Dataset split.

    Returns:
        List of Episode objects.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets is required for TFDS loading. "
            "Install it separately or use the JSON/HDF5/Parquet loaders."
        ) from exc

    builder = tfds.builder(dataset_name)
    builder.download_and_prepare()
    dataset = builder.as_dataset(split=split)
    episodes: list[Episode] = []
    for record in tfds.as_numpy(dataset):
        if not isinstance(record, dict):
            raise ValueError(f"Expected TFDS record to be a mapping, got {type(record)!r}")
        episodes.append(Episode(EpisodeSchema.model_validate(record)))
    return episodes
