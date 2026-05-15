"""Tests for tactile sensors."""

from __future__ import annotations

import torch
from tfwm.sim.sensors import BioTacSensor, GelSightSensor, TactileSensorConfig


class TestTactileSensors:
    """Test tactile sensor implementations."""

    def test_gelsight_sensor(self):
        """Test GelSight sensor."""
        config = TactileSensorConfig(
            site_name="test_site",
            n_taxels=(5, 5),
            taxel_spacing=0.001,
        )
        sensor = GelSightSensor(config)

        # Test initial reading
        reading = sensor.get_reading()
        assert reading.shape == (5, 5)
        assert torch.all(reading == 0.0)

        # Test reset
        sensor.reset()
        reading = sensor.get_reading()
        assert torch.all(reading == 0.0)

    def test_biotac_sensor(self):
        """Test BioTac sensor."""
        config = TactileSensorConfig(
            site_name="test_site",
            n_taxels=(4, 4),
            taxel_spacing=0.001,
        )
        sensor = BioTacSensor(config)

        # Test initial reading
        reading = sensor.get_reading()
        assert reading.shape == (4, 4)
        assert torch.all(reading == 0.5)

        # Test reset
        sensor.reset()
        reading = sensor.get_reading()
        assert torch.all(reading == 0.5)

    def test_sensor_config(self):
        """Test sensor configuration."""
        config = TactileSensorConfig(
            site_name="finger",
            n_taxels=(10, 20),
            taxel_spacing=0.002,
            contact_threshold=0.01,
        )

        assert config.site_name == "finger"
        assert config.n_taxels == (10, 20)
        assert config.taxel_spacing == 0.002
        assert config.contact_threshold == 0.01
