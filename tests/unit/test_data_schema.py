"""Tests for data schema."""


from tfwm.data.schema import ActionData, Episode, EpisodeMetadata, SensorData


def test_sensor_data():
    """Test SensorData model."""
    data = SensorData(
        tactile=[1.0, 2.0, 3.0],
        vision=None,
        proprio=[0.1, 0.2],
        contact_events=[0, 1, 0],
        timestamp=1.0,
    )
    assert data.tactile == [1.0, 2.0, 3.0]
    assert data.vision is None


def test_action_data():
    """Test ActionData model."""
    action = ActionData(action=[0.5, -0.2], timestamp=1.5)
    assert action.action == [0.5, -0.2]


def test_episode_metadata():
    """Test EpisodeMetadata model."""
    meta = EpisodeMetadata(
        task_name="inhand_reorient",
        sim_name="mujoco",
        seed=42,
        randomization_seed=123,
        success=True,
        steps=100,
        duration_s=10.5,
        camera_queries=5,
        slip_events=2,
        force_violations=0,
    )
    assert meta.task_name == "inhand_reorient"
    assert meta.success is True


def test_episode():
    """Test Episode model."""
    obs = SensorData(tactile=[1.0], timestamp=0.0)
    action = ActionData(action=[0.0], timestamp=0.0)
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

    episode = Episode(
        metadata=meta,
        observations=[obs],
        actions=[action],
        rewards=[1.0],
        dones=[True],
    )

    assert len(episode) == 1
    assert episode.n_steps == 1
