"""Parquet data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..episode import Episode
from ..schema import Episode as EpisodeSchema


def load_episodes_from_parquet(path: Path) -> list[Episode]:
    """Load episodes from Parquet file.

    Args:
        path: Path to Parquet file.

    Returns:
        List of Episode objects.
    """
    table = pq.read_table(path)
    frame = table.to_pandas()
    episodes = []
    for record in frame["episode_json"]:
        schema = EpisodeSchema.model_validate_json(record)
        episodes.append(Episode(schema))
    return episodes


def save_episodes_to_parquet(episodes: list[Episode], path: Path) -> None:
    """Save episodes to Parquet file.

    Args:
        episodes: List of episodes to save.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "episode_index": list(range(len(episodes))),
            "episode_json": [episode.schema.model_dump_json() for episode in episodes],
        }
    )
    frame.to_parquet(path, index=False)
