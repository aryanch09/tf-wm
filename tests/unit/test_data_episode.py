"""Tests for episode handling."""

import tempfile
from pathlib import Path

from tfwm.data.episode import Episode
from tfwm.data.schema import ActionData, EpisodeMetadata, SensorData
from tfwm.data.schema import Episode as EpisodeSchema


def test_episode_from_schema():
    """Test creating Episode from schema."""
    meta = EpisodeMetadata(
        task_name="test",
        sim_name="test",
        seed=1,
        randomization_seed=1,
        success=True,
        steps=2,
        duration_s=2.0,
        camera_queries=0,
        slip_events=0,
        force_violations=0,
    )

    obs1 = SensorData(tactile=[1.0, 2.0], timestamp=0.0)
    obs2 = SensorData(tactile=[3.0, 4.0], timestamp=1.0)
    action1 = ActionData(action=[0.5], timestamp=0.0)
    action2 = ActionData(action=[-0.5], timestamp=1.0)

    schema = EpisodeSchema(
        metadata=meta,
        observations=[obs1, obs2],
        actions=[action1, action2],
        rewards=[0.0, 1.0],
        dones=[False, True],
    )

    episode = Episode(schema)
    assert len(episode) == 2


def test_episode_to_tensors():
    """Test converting episode to tensors."""
    # Create minimal episode
    meta = EpisodeMetadata(
        task_name="test",
        sim_name="test",
        seed=1,
        randomization_seed=1,
        success=True,
        steps=1,
        duration_s=1.0,
        camera_queries=0,
        slip_events=0,
        force_violations=0,
    )

    obs = SensorData(tactile=[1.0], timestamp=0.0)
    action = ActionData(action=[0.5], timestamp=0.0)

    schema = EpisodeSchema(
        metadata=meta,
        observations=[obs],
        actions=[action],
        rewards=[1.0],
        dones=[True],
    )

    episode = Episode(schema)
    tensors = episode.to_tensors()

    assert "tactile" in tensors
    assert "actions" in tensors
    assert tensors["tactile"].shape == (1, 1, 1)  # (T, N_taxel, C)
    assert tensors["actions"].shape == (1, 1)  # (T, d_a)


def test_episode_to_tensors_pads_variable_width_fields():
    """Test robust tensor conversion for variable-width observations."""
    meta = EpisodeMetadata(
        task_name="test",
        sim_name="test",
        seed=1,
        randomization_seed=1,
        success=True,
        steps=2,
        duration_s=1.0,
        camera_queries=0,
        slip_events=0,
        force_violations=0,
    )
    schema = EpisodeSchema(
        metadata=meta,
        observations=[
            SensorData(tactile=[1.0], proprio=[0.1], vision=[0.5], timestamp=0.0),
            SensorData(tactile=[1.0, 2.0, 3.0], proprio=[0.2, 0.3], timestamp=1.0),
        ],
        actions=[
            ActionData(action=[0.5], timestamp=0.0),
            ActionData(action=[0.5, 0.6], timestamp=1.0),
        ],
        rewards=[1.0, 2.0],
        dones=[False, True],
    )

    tensors = Episode(schema).to_tensors()

    assert tensors["tactile"].shape == (2, 3, 1)
    assert tensors["proprio"].shape == (2, 2)
    assert tensors["actions"].shape == (2, 2)
    assert tensors["vision"].shape == (2, 3, 64, 64)
    assert tensors["vision"][0, 0, 0, 0] == 0.5


def test_episode_json_roundtrip():
    """Test saving and loading episode from JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode.json"

        # Create episode
        meta = EpisodeMetadata(
            task_name="test",
            sim_name="test",
            seed=1,
            randomization_seed=1,
            success=True,
            steps=1,
            duration_s=1.0,
            camera_queries=0,
            slip_events=0,
            force_violations=0,
        )

        obs = SensorData(tactile=[1.0], timestamp=0.0)
        action = ActionData(action=[0.5], timestamp=0.0)

        schema = EpisodeSchema(
            metadata=meta,
            observations=[obs],
            actions=[action],
            rewards=[1.0],
            dones=[True],
        )

        original = Episode(schema)
        original.to_json(path)

        loaded = Episode.from_json(path)
        assert len(loaded) == len(original)
        assert loaded.metadata.task_name == original.metadata.task_name


def test_episode_parquet_roundtrip(tmp_path):
    """Test saving and loading a single episode from Parquet."""
    meta = EpisodeMetadata(
        task_name="test",
        sim_name="test",
        seed=1,
        randomization_seed=1,
        success=True,
        steps=1,
        duration_s=1.0,
        camera_queries=0,
        slip_events=0,
        force_violations=0,
    )
    schema = EpisodeSchema(
        metadata=meta,
        observations=[SensorData(tactile=[1.0], timestamp=0.0)],
        actions=[ActionData(action=[0.5], timestamp=0.0)],
        rewards=[1.0],
        dones=[True],
    )
    path = tmp_path / "episode.parquet"

    Episode(schema).to_parquet(path)
    loaded = Episode.from_parquet(path)

    assert loaded.metadata.task_name == "test"
    assert len(loaded) == 1
