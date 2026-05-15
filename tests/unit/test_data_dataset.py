"""Tests for dataset classes."""

import tempfile
from pathlib import Path

from tfwm.data.dataset import DirectoryDataset, EpisodeDataset, SequenceDataset
from tfwm.data.episode import Episode
from tfwm.data.schema import ActionData, EpisodeMetadata, SensorData
from tfwm.data.schema import Episode as EpisodeSchema


def create_test_episode(length: int = 10) -> Episode:
    """Create a test episode."""
    meta = EpisodeMetadata(
        task_name="test",
        sim_name="test",
        seed=1,
        randomization_seed=1,
        success=True,
        steps=length,
        duration_s=float(length),
        camera_queries=0,
        slip_events=0,
        force_violations=0,
    )

    observations = []
    actions = []
    for i in range(length):
        obs = SensorData(tactile=[float(i)], timestamp=float(i))
        observations.append(obs)
        action = ActionData(action=[0.1 * i], timestamp=float(i))
        actions.append(action)

    schema = EpisodeSchema(
        metadata=meta,
        observations=observations,
        actions=actions,
        rewards=[1.0] * length,
        dones=[False] * (length - 1) + [True],
    )

    return Episode(schema)


def test_episode_dataset():
    """Test EpisodeDataset."""
    episodes = [create_test_episode(5), create_test_episode(3)]
    dataset = EpisodeDataset(episodes)

    assert len(dataset) == 2

    item = dataset[0]
    assert "tactile" in item
    assert "actions" in item


def test_sequence_dataset():
    """Test SequenceDataset."""
    episodes = [create_test_episode(20)]
    dataset = SequenceDataset(episodes, seq_length=5, stride=2)

    assert len(dataset) > 0

    item = dataset[0]
    assert item["tactile"].shape[0] == 5  # Sequence length


def test_directory_dataset():
    """Test DirectoryDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create test episode files
        episode = create_test_episode(5)
        episode.to_json(data_dir / "ep1.json")
        episode.to_json(data_dir / "ep2.json")

        dataset = DirectoryDataset(data_dir)
        assert len(dataset) == 2

        item = dataset[0]
        assert "tactile" in item
