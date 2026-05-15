"""Tactile spectrogram visualisation with slip-probability overlay."""

from __future__ import annotations

import numpy as np


def plot_tactile_spectrogram(
    pred: "np.ndarray",
    gt: "np.ndarray",
    slip_probs: "np.ndarray | None" = None,
    taxel_idx: int = 0,
    channel_idx: int = 0,
    title: str = "Tactile Spectrogram",
) -> "matplotlib.figure.Figure":
    """Plot predicted vs. ground-truth tactile frequency content.

    Computes short-time Fourier transform (STFT) along the time axis for a
    single taxel/channel and renders side-by-side spectrograms.  If
    ``slip_probs`` is provided, overlays a slip probability curve.

    Args:
        pred: Predicted tactile ``(T, N, C)`` or ``(T,)``.
        gt: Ground-truth tactile, same shape as ``pred``.
        slip_probs: Optional slip probability array ``(T,)`` in ``[0, 1]``.
        taxel_idx: Taxel index to display.
        channel_idx: Channel index to display.
        title: Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram  # type: ignore[import]

    def _extract(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 3:
            return arr[:, taxel_idx, channel_idx]
        return arr.squeeze()

    p_sig = _extract(pred)
    g_sig = _extract(gt)
    T = len(p_sig)
    fs = 1.0  # normalised frequency axis
    nperseg = min(32, T // 4 or 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=12)

    for row, (sig, name) in enumerate([(g_sig, "GT"), (p_sig, "Pred")]):
        # Time domain.
        axes[row, 0].plot(sig, lw=1.0, color="steelblue" if name == "GT" else "orange")
        axes[row, 0].set_title(f"{name} (time)")
        axes[row, 0].set_xlabel("t")
        # Spectrogram.
        f, t_seg, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg)
        axes[row, 1].pcolormesh(t_seg, f, 10 * np.log10(Sxx.clip(1e-10)), shading="auto")
        axes[row, 1].set_title(f"{name} (spectrogram)")
        axes[row, 1].set_xlabel("t")
        axes[row, 1].set_ylabel("Freq")

    if slip_probs is not None:
        for col in (0, 1):
            ax2 = axes[0, col].twinx()
            ax2.plot(slip_probs, color="red", alpha=0.5, lw=1.0, ls="--", label="P(slip)")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("P(slip)", color="red", fontsize=7)

    fig.tight_layout()
    return fig
