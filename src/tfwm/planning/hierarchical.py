"""Hierarchical planner: affordance graph → sub-goal → low-level MPC.

The high-level planner selects the next sub-goal from the affordance graph;
the low-level planner (CEM-MPC or MPPI) generates a sequence of actions to
reach that sub-goal within a short horizon.
"""

from __future__ import annotations

import torch

from tfwm.planning.affordance_graph import AffordanceGraph
from tfwm.planning.cem_mpc import CEMMPCPlanner, CEMMPCConfig
from tfwm.errors import ShapeError


class HierarchicalPlanner:
    """Two-level hierarchical planner.

    Args:
        affordance_graph: High-level node graph.
        low_level_planner: :class:`~tfwm.planning.cem_mpc.CEMMPCPlanner`
            (or any planner with a ``plan(z_t, goal, context)`` method).
        replan_every: Re-query the affordance graph every N steps.

    Shape:
        - ``z_t``:      ``(B, 1, d_z)`` or ``(B, d_z)``
        - ``goal_node``: int — target affordance node index
        - Returns action: ``(B, d_a)``
    """

    def __init__(
        self,
        affordance_graph: AffordanceGraph,
        low_level_planner: CEMMPCPlanner,
        replan_every: int = 5,
    ) -> None:
        self.graph = affordance_graph
        self.planner = low_level_planner
        self.replan_every = replan_every
        self._step_counter: int = 0
        self._current_subgoal: torch.Tensor | None = None

    def plan(
        self,
        z_t: torch.Tensor,
        goal_node: int,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate one action toward ``goal_node``.

        Args:
            z_t: Current latent ``(B, d_z)`` or ``(B, 1, d_z)``.
            goal_node: Target affordance node index.
            context: Optional context tensor.

        Returns:
            Action ``(B, d_a)``.
        """
        if z_t.dim() == 2:
            z_t_flat = z_t
        elif z_t.dim() == 3:
            z_t_flat = z_t[:, -1]
        else:
            raise ShapeError(f"Expected z_t of rank 2 or 3, got {z_t.dim()}")

        if self._step_counter % self.replan_every == 0 or self._current_subgoal is None:
            self._current_subgoal = self.graph(z_t_flat, goal_node)

        self._step_counter += 1
        return self.planner.plan(z_t, self._current_subgoal, context)

    def reset(self) -> None:
        """Reset step counter and cached sub-goal (call at episode start)."""
        self._step_counter = 0
        self._current_subgoal = None
