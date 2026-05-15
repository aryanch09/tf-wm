import torch
from tfwm.models.vision import VisionDecoder, VisionEncoder


def test_vision_encoder_decoder_roundtrip() -> None:
    encoder = VisionEncoder(input_channels=3, latent_dim=32)
    decoder = VisionDecoder(latent_dim=32, output_channels=3)

    image = torch.randn(1, 3, 64, 64)
    latent = encoder(image)
    reconstruction = decoder(latent)

    assert reconstruction.shape == image.shape
    assert latent.shape == (1, 32)
