"""Gate score timeline visualisation."""

from __future__ import annotations

import numpy as np


def plot_gating_timeline(
    gate_scores: "np.ndarray | list[float]",
    queries: "np.ndarray | list[bool]",
    success_marker: int | None = None,
    threshold: float = 0.5,
    title: str = "VOI Gate Scores",
) -> "matplotlib.figure.Figure":
    """Plot gate scores over time with vision-query markers.

    Args:
        gate_scores: Gate output in ``[0, 1]`` of shape ``(T,)``.
        queries: Boolean array ``(T,)`` — ``True`` when vision was queried.
        success_marker: Timestep index where success was achieved, or ``None``.
        threshold: Gate decision threshold (dashed line).
        title: Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(gate_scores, dtype=np.float32)
    q_mask = np.asarray(queries, dtype=bool)
    T = len(scores)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(scores, color="steelblue", lw=1.5, label="Gate score")
    ax.axhline(threshold, color="gray", ls="--", lw=1.0, label=f"Threshold {threshold:.2f}")

    q_idx = np.where(q_mask)[0]
    ax.scatter(q_idx, scores[q_idx], color="orange", zorder=5, s=30, label="Vision queried")

    if success_marker is not None:
        ax.axvline(success_marker, color="green", ls=":", lw=2.0, label="Success")

    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gate score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
