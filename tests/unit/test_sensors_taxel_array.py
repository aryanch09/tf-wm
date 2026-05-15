"""Tests for taxel-array sensors."""

from __future__ import annotations

import pytest
import torch
from tfwm.errors import ShapeError
from tfwm.sensors import TaxelArraySensor, TaxelArraySensorConfig


def test_taxel_array_sensor_read_and_calibrate() -> None:
    sensor = TaxelArraySensor(TaxelArraySensorConfig(n_taxels=4, n_channels=2, baseline=0.1))
    assert sensor.read().shape == (1, 1, 4, 2)

    sensor.set_reading(torch.ones(4, 2))
    assert torch.all(sensor.read() == 1.0)

    sensor.calibrate()
    assert torch.allclose(sensor.read(), torch.full((1, 1, 4, 2), 0.1))


def test_taxel_array_sensor_rejects_bad_shape() -> None:
    sensor = TaxelArraySensor(TaxelArraySensorConfig(n_taxels=4, n_channels=1))
    with pytest.raises(ShapeError):
        sensor.set_reading(torch.ones(3, 2))
