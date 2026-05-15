"""Tests for simulation environment."""

from __future__ import annotations

from tfwm.sim.sensors import BioTacSensor, GelSightSensor, TactileSensorConfig


class TestTFWMSimEnv:
    """Test simulation environment."""

    def test_env_initialization(self):
        """Test environment initialization."""
        # This would require a MuJoCo model file
        # For now, just test that the class can be instantiated

    def test_sensor_integration(self):
        """Test sensor integration."""
        # Test sensor configurations
        config1 = TactileSensorConfig(
            site_name="finger1",
            n_taxels=(10, 10),
            taxel_spacing=0.001,
        )
        config2 = TactileSensorConfig(
            site_name="finger2",
            n_taxels=(8, 8),
            taxel_spacing=0.001,
        )

        sensor1 = GelSightSensor(config1)
        sensor2 = BioTacSensor(config2)

        assert sensor1.config.n_taxels == (10, 10)
        assert sensor2.config.n_taxels == (8, 8)

        # Test sensor readings
        reading1 = sensor1.get_reading()
        reading2 = sensor2.get_reading()

        assert reading1.shape == (10, 10)
        assert reading2.shape == (8, 8)
