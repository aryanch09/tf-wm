"""Domain randomisation sampler for sim-to-real transfer.

Randomises physics parameters (friction, mass, object scale), sensor
properties (calibration offsets, taxel dropout, sampling jitter), and
rendering properties (lighting, textures) at episode reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomisationConfig:
    """Ranges for each randomised parameter."""

    # Physics
    friction_range: tuple[float, float] = (0.3, 1.2)
    mass_scale_range: tuple[float, float] = (0.7, 1.3)
    object_scale_range: tuple[float, float] = (0.9, 1.1)

    # Tactile sensor
    taxel_noise_std_range: tuple[float, float] = (0.001, 0.05)
    taxel_dropout_prob: float = 0.05
    calibration_offset_std: float = 0.02
    sampling_jitter_ms: float = 2.0

    # Rendering (applied in sim only)
    lighting_intensity_range: tuple[float, float] = (0.5, 1.5)
    texture_variant_count: int = 8


class DomainRandomiser:
    """Sample a new randomisation at episode start.

    Args:
        cfg: :class:`RandomisationConfig` with parameter ranges.
        seed: RNG seed.

    Usage::

        rand = DomainRandomiser()
        params = rand.sample()
        env.apply_randomisation(params)
    """

    def __init__(self, cfg: RandomisationConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or RandomisationConfig()
        self._rng = np.random.default_rng(seed)

    def sample(self) -> dict[str, float | int | np.ndarray]:
        """Draw one randomisation dict.

        Returns:
            Dict with keys:
            ``friction``, ``mass_scale``, ``object_scale``,
            ``taxel_noise_std``, ``taxel_dropout_mask``,
            ``calibration_offsets``, ``lighting_intensity``,
            ``texture_id``.
        """
        cfg = self.cfg
        n_taxels_placeholder = 16  # will be overridden by caller if needed

        friction = float(self._rng.uniform(*cfg.friction_range))
        mass_scale = float(self._rng.uniform(*cfg.mass_scale_range))
        object_scale = float(self._rng.uniform(*cfg.object_scale_range))
        taxel_noise_std = float(self._rng.uniform(*cfg.taxel_noise_std_range))
        dropout_mask = (self._rng.random(n_taxels_placeholder) > cfg.taxel_dropout_prob).astype(np.float32)
        calib_offsets = self._rng.normal(0.0, cfg.calibration_offset_std, n_taxels_placeholder).astype(np.float32)
        lighting = float(self._rng.uniform(*cfg.lighting_intensity_range))
        texture_id = int(self._rng.integers(0, cfg.texture_variant_count))

        return {
            "friction": friction,
            "mass_scale": mass_scale,
            "object_scale": object_scale,
            "taxel_noise_std": taxel_noise_std,
            "taxel_dropout_mask": dropout_mask,
            "calibration_offsets": calib_offsets,
            "lighting_intensity": lighting,
            "texture_id": texture_id,
            "sampling_jitter_ms": cfg.sampling_jitter_ms * self._rng.random(),
        }

    def apply_tactile_noise(
        self,
        tactile: np.ndarray,
        params: dict[str, float | int | np.ndarray],
    ) -> np.ndarray:
        """Apply noise model to a raw tactile reading.

        Args:
            tactile: Raw tactile array ``(N, C)``.
            params: Randomisation params from :meth:`sample`.

        Returns:
            Noisified tactile ``(N, C)``.
        """
        N, C = tactile.shape
        noise_std = float(params["taxel_noise_std"])
        dropout = np.asarray(params["taxel_dropout_mask"]).reshape(-1, 1)[:N]
        calib = np.asarray(params["calibration_offsets"]).reshape(-1, 1)[:N]

        out = tactile + self._rng.normal(0.0, noise_std, tactile.shape).astype(np.float32)
        out = out * dropout + calib
        return np.clip(out, 0.0, 1.0)
