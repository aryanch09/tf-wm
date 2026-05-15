"""Tests for shape utilities."""

import torch
from tfwm.utils.shape import batch_repeat, pack_sequences, temporal_crop, unpack_sequences


def test_pack_sequences():
    """Test packing variable-length sequences."""
    seq1 = torch.randn(5, 3)
    seq2 = torch.randn(3, 3)
    seq3 = torch.randn(7, 3)

    packed, lengths = pack_sequences([seq1, seq2, seq3])

    assert packed.shape == (3, 7, 3)
    assert torch.equal(lengths, torch.tensor([5, 3, 7]))

    # Check padding
    assert torch.allclose(packed[0, 5:], torch.zeros(2, 3))
    assert torch.allclose(packed[1, 3:], torch.zeros(4, 3))


def test_unpack_sequences():
    """Test unpacking sequences."""
    packed = torch.randn(3, 7, 3)
    lengths = torch.tensor([5, 3, 7])

    unpacked = unpack_sequences(packed, lengths)

    assert len(unpacked) == 3
    assert unpacked[0].shape == (5, 3)
    assert unpacked[1].shape == (3, 3)
    assert unpacked[2].shape == (7, 3)


def test_temporal_crop():
    """Test temporal cropping."""
    tensor = torch.randn(2, 10, 3)
    cropped = temporal_crop(tensor, start=2, length=5)

    assert cropped.shape == (2, 5, 3)
    assert torch.equal(cropped, tensor[:, 2:7])


def test_batch_repeat():
    """Test batch repetition."""
    tensor = torch.randn(3, 4)
    repeated = batch_repeat(tensor, 2)

    assert repeated.shape == (6, 4)
    assert torch.equal(repeated[:3], tensor)
    assert torch.equal(repeated[3:], tensor)
