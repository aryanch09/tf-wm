"""UMAP projection of latent tactile representations."""

from __future__ import annotations

import numpy as np


def plot_latent_umap(
    latents: "np.ndarray",
    labels: "np.ndarray | list[int] | None" = None,
    label_names: list[str] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "Latent UMAP",
) -> "matplotlib.figure.Figure":
    """Embed latents with UMAP and produce a scatter plot.

    Args:
        latents: ``(N, d_z)`` array of latent vectors.
        labels: ``(N,)`` integer class labels for colouring.
        label_names: Human-readable name per class index.
        n_neighbors: UMAP ``n_neighbors`` parameter.
        min_dist: UMAP ``min_dist`` parameter.
        title: Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.

    Raises:
        ImportError: If ``umap-learn`` is not installed (gracefully falls back
            to PCA for CI environments).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import umap  # type: ignore[import]
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(latents)
    except ImportError:
        from sklearn.decomposition import PCA  # type: ignore[import]
        embedding = PCA(n_components=2, random_state=42).fit_transform(latents)

    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is not None:
        lbl = np.asarray(labels)
        for cls in np.unique(lbl):
            mask = lbl == cls
            name = label_names[int(cls)] if label_names else str(cls)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.7, label=name)
        ax.legend(fontsize=8, markerscale=2)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    return fig
