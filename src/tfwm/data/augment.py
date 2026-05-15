"""Data augmentation for tactile and vision."""

from __future__ import annotations

import torch
from torchvision import transforms

from ..types import Tactile, Vision


def augment_tactile(tactile: Tactile, noise_std: float = 0.01) -> Tactile:
    """Augment tactile data with noise.

    Args:
        tactile: Tactile tensor (B, T, N_taxel, C).
        noise_std: Standard deviation of additive noise.

    Returns:
        Augmented tactile tensor.
    """
    noise = torch.randn_like(tactile) * noise_std
    return tactile + noise


def augment_vision(vision: Vision) -> Vision:
    """Augment vision data with standard transforms.

    Args:
        vision: Vision tensor (B, T, C, H, W).

    Returns:
        Augmented vision tensor.
    """
    # Apply torchvision transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    )

    # Apply to each frame
    B, T, C, H, W = vision.shape
    augmented = vision.clone()

    for b in range(B):
        for t in range(T):
            frame = vision[b, t]  # (C, H, W)
            # Convert to PIL-like tensor for transforms
            frame = transform(frame)
            augmented[b, t] = frame

    return augmented


class TactileAugmenter:
    """Tactile data augmentation pipeline."""

    def __init__(self, noise_std: float = 0.01, dropout_prob: float = 0.05):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __call__(self, tactile: Tactile) -> Tactile:
        """Apply augmentation.

        Args:
            tactile: Input tactile tensor.

        Returns:
            Augmented tensor.
        """
        # Add noise
        augmented = augment_tactile(tactile, self.noise_std)

        # Taxel dropout
        mask = torch.rand_like(augmented[..., 0:1]) > self.dropout_prob
        augmented = augmented * mask

        return augmented


class VisionAugmenter:
    """Vision data augmentation pipeline."""

    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
            ]
        )

    def __call__(self, vision: Vision) -> Vision:
        """Apply augmentation.

        Args:
            vision: Input vision tensor.

        Returns:
            Augmented tensor.
        """
        return augment_vision(vision)
