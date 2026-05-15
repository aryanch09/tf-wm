"""HDF5 data loader."""

from __future__ import annotations

from pathlib import Path

import h5py

from ..episode import Episode
from ..schema import Episode as EpisodeSchema


def load_episodes_from_hdf5(path: Path) -> list[Episode]:
    """Load episodes from HDF5 file.

    Args:
        path: Path to HDF5 file.

    Returns:
        List of Episode objects.
    """
    episodes = []
    with h5py.File(path, "r") as handle:
        group = handle["episodes"]
        for key in sorted(group.keys()):
            raw = group[key][()]
            if isinstance(raw, bytes):
                payload = raw.decode("utf-8")
            else:
                payload = raw.tobytes().decode("utf-8")
            episodes.append(Episode(EpisodeSchema.model_validate_json(payload)))
    return episodes


def save_episodes_to_hdf5(episodes: list[Episode], path: Path) -> None:
    """Save episodes to HDF5 file.

    Args:
        episodes: List of episodes to save.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("episodes")
        for index, episode in enumerate(episodes):
            group.create_dataset(
                f"episode_{index:05d}",
                data=episode.schema.model_dump_json().encode("utf-8"),
            )
