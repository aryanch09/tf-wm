"""Tests for tactile models."""

from __future__ import annotations

import torch
from tfwm.models.tactile import TactileDecoder, TactileEncoder


class TestTactileModels:
    def test_tactile_encoder(self):
        encoder = TactileEncoder(d_latent=64)
        tactile = torch.ones((1, 16, 16), dtype=torch.float32)
        latent = encoder(tactile)
        assert latent.shape == (1, 64)

    def test_tactile_encoder_batch(self):
        encoder = TactileEncoder(d_latent=64)
        tactile = torch.ones((4, 16, 16), dtype=torch.float32)
        latent = encoder(tactile)
        assert latent.shape == (4, 64)

    def test_tactile_decoder(self):
        decoder = TactileDecoder(output_shape=(16, 16))
        latent = torch.ones((1, 64), dtype=torch.float32)
        reconstructed = decoder(latent)
        assert reconstructed.shape == (1, 16, 16)

    def test_tactile_decoder_batch(self):
        decoder = TactileDecoder(output_shape=(16, 16))
        latent = torch.ones((4, 64), dtype=torch.float32)
        reconstructed = decoder(latent)
        assert reconstructed.shape == (4, 16, 16)

    def test_autoencoder_consistency(self):
        encoder = TactileEncoder(d_latent=64)
        decoder = TactileDecoder(output_shape=(16, 16))
        tactile = torch.ones((1, 16, 16), dtype=torch.float32)
        latent = encoder(tactile)
        reconstructed = decoder(latent)
        assert reconstructed.shape == tactile.shape
