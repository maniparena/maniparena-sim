# ManipArena-Sim Installation (Lab 3 / Sim 6)

## Design

Approach **A** (approved): consume unmodified [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) as a git submodule; Isaac Lab comes from Arena's nested `submodules/IsaacLab`; Isaac Sim is installed as **binary wheels** (no Sim source build).

Default install is a **lean sim stack** for teleop collection and ROS2 navigation. It does **not** use Arena's fat `uv sync` default (`isaaclab-from-source` + `openpi`).

| Component | Location | Pin |
|-----------|----------|-----|
| IsaacLab-Arena | `3rd/isaaclabarena` | GitHub `main` |
| Isaac Lab | `3rd/isaaclabarena/submodules/IsaacLab` | Arena submodule SHA (Lab 3.0; not `develop` tip) |
| Isaac Sim | wheels `isaacsim[all,extscache]==6.0.1.0` | 6.0.x |

Lab **must** stay on Arena's recorded SHA. `develop` tip moved `isaaclab_tasks.manager_based.*` → `contrib.*`, which breaks Arena `main` imports (e.g. Agibot). Override only if you know you need it: `source ./install.sh --lab-branch develop`.

**Offline / black viewport (EX001):** all USD asset refs must stay under `assets/`
(no `omniverse://`, S3, or `../` MDL module paths). EX001 booth marble is vendored at
`assets/green_booth/materials/stone/`; the 2D lidar USD is at
`assets/ex001/sensors/Example_Rotary_2D.usda`. Keep OCIO tonemap enabled in
`assets/rendering_settings/green_booth.settings.usda`
(`rtx:post:tonemap:ocio:enabled = 1`); turning it off causes an overexposed viewport.

The installer also applies lean Arena import patches so EX001 teleop/nav does not need the fat multi-robot / RL stack:

- `embodiments/__init__.py` → no-op (skip agibot/g1/... / onnxruntime)
- `policy/__init__.py` → skip optional `rsl_rl_action_policy` (needs `rsl-rl-lib`)
- `assets/registries.ensure_assets_registered` → soft-import optional libraries

Explicit imports (e.g. `embodiment_base`) still work. Use `--full-arena-stack` only if you need upstream Arena RL/policy extras.
| ManipArena-Sim | repo root | `uv pip install -e .` |

## What gets installed (default)

```text
isaacsim[all,extscache]     # Sim 6 wheels (explicit)
isaaclab                    # Lab core (editable; no [isaacsim] extra)
isaaclab_assets / physx / newton / ov / ovphysx
isaaclab_tasks              # ArenaEnvBuilder → parse_env_cfg
isaaclab_teleop
isaaclab_visualizers[kit]   # required for Lab 6 `--viz kit` viewport
isaaclab_arena              # --no-deps + thin runtime deps
maniparena_sim

# Undeclared-but-required runtime wheels (lean install adds these):
hydra-core + omegaconf      # isaaclab_tasks.utils.parse_env_cfg / ArenaEnvBuilder
pyyaml, filelock, pillow, h5py, gymnasium, trimesh, einops, ...
pyarrow                       # LeRobot parquet export (H-key success save)
vuer, pydantic, scipy, pandas  # Arena teleop / config types
```

## Explicitly excluded

| Skipped | Why |
|---------|-----|
| `openpi` / `openpi-client` | Policy client; not used by collect/nav |
| GR00T / DreamZero / Cosmos Arena extras | Policy backends |
| `isaaclab_mimic` | Imitation-learning tooling |
| `isaaclab_rl` / RSL-RL stack | RL training |
| `isaaclab_visualizers[newton/rerun/viser]` | Non-Kit visualizer backends |
| `isaaclab_tasks_experimental` / `contrib` / `experimental` / `ppisp` | Unused by ManipArena |
| Arena hard extras: `openai`, `sbi`, `onnxruntime`, `pytest`, `lightwheel-sdk` | Analysis / asset CDN; ManipArena uses local USD |

Escape hatch (not recommended): `source ./install.sh --full-arena-stack`.

## One-click

```bash
git lfs install
git clone --recurse-submodules https://github.com/maniparena/maniparena-sim.git
cd maniparena-sim
source ./install.sh
source 3rd/isaaclabarena/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
```

If a previous fat `uv sync` already started, stop it (`Ctrl+C`), remove the half-built env, and re-run the lean installer:

```bash
rm -rf 3rd/isaaclabarena/.venv
source ./install.sh
```

See root `README.md` for prerequisites and ROS2 notes.
