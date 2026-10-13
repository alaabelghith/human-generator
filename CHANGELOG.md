# Changelog

## 1.0.0 (2026-10-13)

First public release.

- One Python SOP (`Human Generator`) for Houdini 20.5.
- Anny body in real units: age (years), height (cm) and weight (kg) solved exactly; gender, muscle, proportions, breast size and firmness, buttocks, Anny local changes. SOMA PCA body as an alternative.
- Eyeballs on the eye joints; smooth normals; volume-keeping surface smoothing.
- Text to motion with NVIDIA Kimodo (Kimodo-SOMA-RP-v1): multi-sentence prompts with per-sentence durations and transitions.
- Path following: wire a curve into the node and Kimodo generates the motion on it (Root2D constraint).
- Output: body mesh, KineFX-style skeleton, or both.
- Local server kept warm on the GPU; Houdini starts it automatically.
- One-command Windows installer (`install.ps1`): no compiler needed, pinned versions, Houdini package written for you.
