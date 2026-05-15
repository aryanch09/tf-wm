"""Sensor abstractions and concrete tactile sensor adapters."""

from __future__ import annotations

from tfwm.sensors.base import SensorReading, TactileSensor
from tfwm.sensors.taxel_array import TaxelArraySensor, TaxelArraySensorConfig

__all__ = ["SensorReading", "TactileSensor", "TaxelArraySensor", "TaxelArraySensorConfig"]
