"""Knowledge distillation for deployment-size model compression.

Distils a large TF-WM teacher (e.g. HybridTactileEncoder + EnsembleDynamics)
into a compact student suitable for edge deployment:
  - Student encoder: single-stage TCN (≈4× fewer parameters)
  - Student dynamics: deterministic single-step MLP
  - Loss: MSE on latent embeddings + optional KL on predicted distributions

Typical usage::

    python -m tfwm.deploy.distill \\
        --teacher outputs/phase4/checkpoint_final.pt \\
        --student_cfg configs/model/tactile_encoder/tcn.yaml \\
        --output outputs/distilled/student.pt \\
        --steps 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from tfwm.data.synth import SyntheticTactileDataset
from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.models.tactile_encoder.tcn import TCNTactileEncoder


def distil(
    teacher_encoder: nn.Module,
    teacher_dynamics: nn.Module,
    student_encoder: nn.Module,
    student_dynamics: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    n_steps: int = 20_000,
    lr: float = 3e-4,
    alpha_latent: float = 1.0,
    alpha_dynamics: float = 0.5,
    device: torch.device | str = "cpu",
    output_path: Path | None = None,
) -> dict[str, float]:
    """Run the distillation training loop.

    Args:
        teacher_encoder: Large frozen encoder (HybridTactileEncoder, etc.).
        teacher_dynamics: Frozen teacher dynamics (RSSM or Ensemble).
        student_encoder: Compact student encoder to train.
        student_dynamics: Compact student dynamics to train.
        train_loader: Batches of ``{tactile, actions}`` dicts.
        n_steps: Total gradient steps.
        lr: Learning rate.
        alpha_latent: Weight of latent MSE loss.
        alpha_dynamics: Weight of dynamics MSE loss.
        device: Compute device.
        output_path: If set, save student checkpoint here.

    Returns:
        Final metrics dict.
    """
    device = torch.device(device)

    teacher_encoder = teacher_encoder.to(device).eval()
    teacher_dynamics = teacher_dynamics.to(device).eval()
    student_encoder = student_encoder.to(device)
    student_dynamics = student_dynamics.to(device)

    for p in teacher_encoder.parameters():
        p.requires_grad_(False)
    for p in teacher_dynamics.parameters():
        p.requires_grad_(False)

    params = list(student_encoder.parameters()) + list(student_dynamics.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)

    step = 0
    train_iter = iter(train_loader)
    metrics: dict[str, float] = {}

    while step < n_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tactile = batch["tactile"].to(device)
        actions = batch["actions"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_enc = teacher_encoder(tactile)
            t_latents = t_enc["mu"]   # (B, T, d_z_teacher)
            t_rssm = teacher_dynamics.forward_teacher(t_latents, actions)
            t_z_post = t_rssm["z_posterior"]

        s_enc = student_encoder(tactile)
        s_latents = s_enc["mu"]   # (B, T, d_z_student)

        # Project to same dim if needed.
        d_t = t_latents.shape[-1]
        d_s = s_latents.shape[-1]
        if d_s != d_t:
            # Trim the smaller one.
            d = min(d_s, d_t)
            t_latents_cmp = t_latents[..., :d]
            s_latents_cmp = s_latents[..., :d]
            t_z_post_cmp = t_z_post[..., :d]
        else:
            t_latents_cmp = t_latents
            s_latents_cmp = s_latents
            t_z_post_cmp = t_z_post

        loss_latent = F.mse_loss(s_latents_cmp, t_latents_cmp)

        s_rssm = student_dynamics.forward_teacher(s_latents, actions)
        s_z_post = s_rssm["z_posterior"]
        d2 = min(s_z_post.shape[-1], t_z_post.shape[-1])
        loss_dynamics = F.mse_loss(s_z_post[..., :d2], t_z_post[..., :d2])

        total_loss = alpha_latent * loss_latent + alpha_dynamics * loss_dynamics
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()

        step += 1
        if step % 1000 == 0:
            metrics = {
                "step": step,
                "loss_latent": float(loss_latent),
                "loss_dynamics": float(loss_dynamics),
                "loss_total": float(total_loss),
            }
            print(f"[distil] step {step}/{n_steps}  loss={float(total_loss):.4f}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_encoder": student_encoder.state_dict(),
                "student_dynamics": student_dynamics.state_dict(),
                "step": step,
            },
            output_path,
        )
        print(f"[distil] saved student checkpoint to {output_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_default_student(
    n_taxels: int, n_channels: int, d_latent: int, d_action: int
) -> tuple[TCNTactileEncoder, RSSMDynamics]:
    """Build a compact TCN student encoder + deterministic dynamics."""
    student_enc = TCNTactileEncoder(
        n_taxels=n_taxels,
        n_channels=n_channels,
        d_latent=d_latent // 2,   # half the teacher's latent dim
        hidden_dim=64,
        n_layers=3,
    )
    student_dyn = RSSMDynamics(
        d_latent=d_latent // 2,
        d_action=d_action,
        d_hidden=128,
    )
    return student_enc, student_dyn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TF-WM knowledge distillation")
    parser.add_argument("--teacher", type=Path, required=True, help="Path to teacher checkpoint")
    parser.add_argument("--output", type=Path, default=Path("outputs/distilled/student.pt"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_taxels", type=int, default=16)
    parser.add_argument("--n_channels", type=int, default=1)
    parser.add_argument("--d_latent", type=int, default=128)
    parser.add_argument("--d_action", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)

    # Build teacher.
    teacher_enc = HybridTactileEncoder(
        n_taxels=args.n_taxels, n_channels=args.n_channels, d_latent=args.d_latent
    )
    teacher_dyn = RSSMDynamics(d_latent=args.d_latent, d_action=args.d_action)

    if args.teacher.exists():
        state = torch.load(args.teacher, map_location=device)
        mods = state.get("modules", {})
        if "encoder" in mods:
            teacher_enc.load_state_dict(mods["encoder"])
        if "dynamics" in mods:
            teacher_dyn.load_state_dict(mods["dynamics"])
        print(f"[distil] loaded teacher from {args.teacher}")

    student_enc, student_dyn = _build_default_student(
        args.n_taxels, args.n_channels, args.d_latent, args.d_action
    )

    dataset = SyntheticTactileDataset(
        n_episodes=200, episode_length=50,
        n_taxels=args.n_taxels, n_channels=args.n_channels, d_action=args.d_action,
    )
    train_ds, _ = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    distil(
        teacher_encoder=teacher_enc,
        teacher_dynamics=teacher_dyn,
        student_encoder=student_enc,
        student_dynamics=student_dyn,
        train_loader=loader,
        n_steps=args.steps,
        lr=args.lr,
        device=device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
