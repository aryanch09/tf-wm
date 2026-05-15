"""Tests for structured episode loaders."""

from __future__ import annotations

from tfwm.data.episode import Episode
from tfwm.data.loaders.hdf5 import load_episodes_from_hdf5, save_episodes_to_hdf5
from tfwm.data.loaders.parquet import load_episodes_from_parquet, save_episodes_to_parquet
from tfwm.data.schema import ActionData, EpisodeMetadata, SensorData
from tfwm.data.schema import Episode as EpisodeSchema


def _episode(seed: int = 1) -> Episode:
    metadata = EpisodeMetadata(
        task_name="loader_test",
        sim_name="synthetic",
        seed=seed,
        randomization_seed=seed,
        success=True,
        steps=1,
        duration_s=0.02,
        camera_queries=0,
        slip_events=0,
        force_violations=0,
    )
    schema = EpisodeSchema(
        metadata=metadata,
        observations=[SensorData(tactile=[0.1, 0.2], timestamp=0.0)],
        actions=[ActionData(action=[0.0], timestamp=0.0)],
        rewards=[1.0],
        dones=[True],
    )
    return Episode(schema)


def test_parquet_loader_roundtrip(tmp_path) -> None:
    path = tmp_path / "episodes.parquet"

    save_episodes_to_parquet([_episode(1), _episode(2)], path)
    episodes = load_episodes_from_parquet(path)

    assert len(episodes) == 2
    assert episodes[0].metadata.task_name == "loader_test"


def test_hdf5_loader_roundtrip(tmp_path) -> None:
    path = tmp_path / "episodes.h5"

    save_episodes_to_hdf5([_episode(1), _episode(2)], path)
    episodes = load_episodes_from_hdf5(path)

    assert len(episodes) == 2
    assert episodes[1].metadata.seed == 2
