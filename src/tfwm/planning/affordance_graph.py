"""Affordance graph for hierarchical planning.

Encodes a directed graph of affordance nodes (grasp, reorient, insert, …)
with learned transition costs in latent space.  The high-level planner
searches this graph to sequence sub-goals; the low-level MPC planner then
reaches each sub-goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class AffordanceNode:
    """A single node in the affordance graph."""

    name: str
    node_id: int
    goal_latent: torch.Tensor | None = None        # (d_z,) — learnt or set from data
    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)


class AffordanceGraph(nn.Module):
    """Differentiable affordance graph with learned edge costs.

    Args:
        n_nodes: Number of affordance nodes.
        d_latent: Latent dimension ``d_z``.
        node_names: Optional list of node names (for logging / visualisation).

    Shape:
        - ``z``: ``(B, d_z)`` — current latent
        - ``goal_node``: integer node index
        - Returns sub-goal latent ``(B, d_z)``.
    """

    def __init__(
        self,
        n_nodes: int = 8,
        d_latent: int = 128,
        node_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.d_latent = d_latent
        self.node_names = node_names or [f"node_{i}" for i in range(n_nodes)]
        # Learnable latent prototype for each node.
        self.node_embeddings = nn.Embedding(n_nodes, d_latent)
        # Learnable adjacency logits (soft edge weights).
        self.edge_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def adjacency(self) -> torch.Tensor:
        """Return soft adjacency matrix ``(N, N)`` via row-wise softmax."""
        return torch.softmax(self.edge_logits, dim=-1)

    def forward(self, z: torch.Tensor, goal_node: int) -> torch.Tensor:
        """Return the sub-goal latent for ``goal_node``.

        Finds the nearest reachable node from the current latent ``z`` toward
        ``goal_node`` using greedy traversal over the soft adjacency matrix.

        Args:
            z: Current latent state ``(B, d_z)``.
            goal_node: Target node index.

        Returns:
            Sub-goal latent ``(B, d_z)``.
        """
        goal_emb = self.node_embeddings.weight[goal_node]  # (d_z,)
        # Find closest node to current z (nearest prototype).
        dists = torch.linalg.vector_norm(
            z.unsqueeze(1) - self.node_embeddings.weight.unsqueeze(0), dim=-1
        )  # (B, N)
        current_node = dists.argmin(dim=-1)  # (B,)
        # Greedy next hop: highest-weight successor toward goal.
        adj = self.adjacency()  # (N, N)
        next_node = adj[current_node, :].argmax(dim=-1)  # (B,)
        return self.node_embeddings(next_node)  # (B, d_z)
