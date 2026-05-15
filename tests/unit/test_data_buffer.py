"""Tests for replay buffer."""

import pytest
from tfwm.data.buffer import PrioritizedReplayBuffer, ReplayBuffer
from tfwm.data.episode import Episode
from tfwm.data.schema import ActionData, EpisodeMetadata, SensorData
from tfwm.data.schema import Episode as EpisodeSchema


def create_test_episode(length: int = 5) -> Episode:
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


def test_replay_buffer_add_and_sample():
    """Test adding episodes and sampling."""
    buffer = ReplayBuffer(capacity=10)

    episode1 = create_test_episode(5)
    episode2 = create_test_episode(3)

    buffer.add_episode(episode1)
    buffer.add_episode(episode2)

    assert len(buffer) == 2

    batch = buffer.sample_batch(2)
    assert len(batch) == 2


def test_replay_buffer_capacity():
    """Test buffer capacity limits."""
    buffer = ReplayBuffer(capacity=2)

    for i in range(3):
        episode = create_test_episode(5)
        buffer.add_episode(episode)

    assert len(buffer) == 2  # Should have removed oldest


def test_prioritized_replay_buffer():
    """Test prioritized replay buffer."""
    buffer = PrioritizedReplayBuffer(capacity=10, alpha=0.6)

    episode1 = create_test_episode(5)
    episode2 = create_test_episode(3)

    buffer.add_episode(episode1, priority=1.0)
    buffer.add_episode(episode2, priority=2.0)

    batch, indices, weights = buffer.sample_batch(2)
    assert len(batch) == 2
    assert len(indices) == 2
    assert len(weights) == 2

    # Update priorities
    buffer.update_priorities(indices, [0.5, 1.5])


def test_buffer_empty_error():
    """Test error when sampling from empty buffer."""
    buffer = ReplayBuffer()

    with pytest.raises(ValueError, match="Buffer is empty"):
        buffer.sample_batch(1)
