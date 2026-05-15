# Changelog

All notable changes to TF-WM are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added — Core Models
- `RSSMDynamics` with teacher-forced `forward_teacher()` and open-loop `rollout()` satisfying the `Dynamics` Protocol (fixes CEM-MPC compatibility)
- `EnsembleDynamics` wrapper with law-of-total-variance epistemic uncertainty
- `TransformerTactileEncoder`: proper causal multi-head self-attention with sinusoidal PE
- `HybridTactileEncoder`: TCN → Transformer two-stage architecture (replaces TCN alias)
- `GatedContextFusion` and `MultiModalFusion` for vision/proprio gating into tactile stream
- `ProprioEncoder` MLP for proprioception embedding
- `TactileDecoder`, `ContactEventHead`, `AffordanceHead`, `ObjectStateHead` decoder heads

### Added — Losses
- `TactileReconLoss` with optional spatial smoothness
- `InfoNCELoss` (CPC) with configurable time/cross-episode negatives
- `ContactBCELoss` with per-class weighting and label smoothing
- `KLDivergenceLoss` with free-nats clipping for RSSM training
- `AffordanceLoss` and `ObjectStateLoss`
- `MultiHorizonLoss` aggregator with geometric discount γ^(h-1) and KL annealing

### Added — Training
- `TrainerBase`: abstract base with AMP, grad clip, W&B logging, checkpoint save/load
- `Phase1Trainer`: full Phase 1 representation learning with teacher-forcing and scheduled sampling
- `build_phase1_trainer()` factory from Hydra config
- Training callbacks: `EMACallback`, `GradClipCallback`, `ScheduledSamplingCallback`, `EarlyStopCallback`
- `SyntheticTactileDataset` for smoke tests without real data

### Added — Planning
- `MPPIPlanner`: single-pass soft-optimal trajectory optimisation
- `SafetyFilter`: CBF-inspired action projection with hard e-stop
- `AffordanceGraph`: differentiable directed graph with learned edge costs
- `HierarchicalPlanner`: two-level affordance graph → CEM-MPC planner

### Added — Policies
- `BCPolicy`: Gaussian behaviour cloning with NLL loss
- `MPCPolicy`: latent-space CEM with optional safety filter
- `SACPolicy` + `TwinCritic`: full SAC actor-critic for Phase 4 RL

### Added — Sim & Sensors
- `InHandReorientEnv`: quaternion-based reorientation with occlusion support
- `PegInHoleEnv`: contact-guided insertion with 80% default occlusion
- `BlindRetrieveEnv`: texture-based blind retrieval
- `OcclusionWrapper` and `OcclusionSweep` for H-A benchmark
- `DomainRandomiser` with physics, tactile, and rendering randomisation
- `DIGITSensor`, `GelSightMiniSensor`, `ReSkinSensor` hardware abstractions
- `RGBCamera` / `RGBDCamera`, `ProprioReader`, `TactileCalibrator`

### Added — Eval
- `AblationRunner` with full 10-variant matrix and Markdown table output
- `RobustnessSweep` for occlusion / noise / sample-rate sweeps
- Visualise: `plot_gating_timeline`, `plot_latent_umap`, `plot_tactile_spectrogram`

### Added — Deploy
- `RealtimeLoop`: three-thread tactile/vision/controller architecture
- `SafetyMonitor`: hard force cutoff and action-norm clipping
- `quantize_dynamic`, `export_torchscript`, `export_onnx` utilities

### Added — Infrastructure
- Complete GitHub Actions CI: lint → tests → smoke-train → CEM latency benchmark
- `dvc.yaml` pipeline with all 5 training phases and eval stage
- Hydra configs: `tcn.yaml`, `transformer.yaml`, `deterministic.yaml`, `ensemble.yaml`,
  `variance.yaml`, `voi.yaml`, `sac.yaml`, `bc.yaml`, `hierarchical.yaml`, `vit_s.yaml`,
  tasks `peg_in_hole`, `blind_retrieve`, `tool_use`,
  evals `ablation`, `robustness`,
  hardware `shadow`, `franka_panda`, `ur5e`,
  data `augmentation`, `domain_rand`,
  train phases 2–5
- Shell scripts: `bootstrap_dev.sh`, `benchmark_realtime.sh`, `export_onnx.sh`, `run_sweep.sh`
- `utils/timing.py`: `Timer` context manager + `TimingStats` accumulator

### Fixed
- `DynamicsModel` missing `rollout()` method (broke CEM-MPC at runtime)
- `TransformerTactileEncoder` was a TCN alias; replaced with causal attention implementation
- `HybridTactileEncoder` was a TCN alias; replaced with TCN→Transformer
- Pyright "Variable not allowed in type expression" errors in dynamics module
- Unused `rearrange` import in dynamics module
