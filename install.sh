#!/usr/bin/env bash
# ManipArena-Sim one-click installer (Approach A, lean sim stack).
#
# Pulls IsaacLab-Arena (GitHub main), pins nested Isaac Lab to the SHA recorded
# by Arena's submodule (Lab 3.0 compatible with Arena imports), then installs a
# *minimal* editable Lab + Isaac Sim wheels + Arena + this package.
#
# Usage:
#   source ./install.sh
#   source ./install.sh --skip-stack            # only prepare git sources
#   source ./install.sh --full-arena-stack      # Arena upstream fat uv sync (NOT recommended)
#   source ./install.sh --lab-branch develop    # bleeding Lab tip (may break Arena)
#
# Default install (sim / teleop / ROS2 nav):
#   IN:  isaaclab(+isaacsim wheels), isaaclab_assets/physx/newton/ov/ovphysx,
#        isaaclab_tasks, isaaclab_teleop, isaaclab_visualizers[kit] (--viz kit),
#        isaaclab_arena (no-deps + thin runtime deps), maniparena_sim
#   OUT: openpi, GR00T, mimic, rl, newton/rerun/viser visualizers,
#        tasks-experimental, contrib, experimental, ppisp,
#        Arena analysis extras (openai/sbi/onnxruntime/...)
#
# Optional env:
#   ISAACSIM_PATH   Existing Isaac Sim 6 install (recorded; primary path is wheels)
#   MANIPARENA_LAB_BRANCH  Override Lab pin (default: arena-pin = Arena submodule SHA)
#   UV_INDEX_URL    Forwarded to uv if set
#
# IMPORTANT: when this file is `source`d, install work runs in a subshell so
# `set -e` / failed pip steps cannot kill your interactive terminal.

_MANIPARENA_INSTALL_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _MANIPARENA_INSTALL_SOURCED=1
fi

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ARENA_REL="3rd/isaaclabarena"
_ARENA_DIR="${_ROOT}/${_ARENA_REL}"
_LAB_REL="${_ARENA_REL}/submodules/IsaacLab"
_LAB_DIR="${_ROOT}/${_LAB_REL}"
_LAB_SRC="${_LAB_DIR}/source"
_ARENA_URL="${MANIPARENA_ARENA_URL:-https://github.com/isaac-sim/IsaacLab-Arena.git}"
_ARENA_BRANCH="${MANIPARENA_ARENA_BRANCH:-main}"
# Default "arena-pin": keep the Lab commit recorded by Arena's submodule.
# Lab develop tip reorganized isaaclab_tasks (manager_based → contrib) and breaks
# Arena main imports such as isaaclab_tasks.manager_based.manipulation.pick_place.
_LAB_BRANCH="${MANIPARENA_LAB_BRANCH:-arena-pin}"
_SKIP_STACK=0
_SKIP_VERIFY=0
_FULL_ARENA_STACK=0

_log() { printf '[maniparena-install] %s\n' "$*"; }
_die() {
  # Always exit the current shell/subshell. Install body runs in a subshell when
  # sourced, so this will not close the user's interactive terminal.
  printf '[maniparena-install] ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-stack|--skip-uv-sync) _SKIP_STACK=1; shift ;;
    --skip-verify) _SKIP_VERIFY=1; shift ;;
    --full-arena-stack) _FULL_ARENA_STACK=1; shift ;;
    --lab-branch)
      [[ $# -ge 2 ]] || _die "--lab-branch requires a value"
      _LAB_BRANCH="$2"
      shift 2
      ;;
    --arena-branch)
      [[ $# -ge 2 ]] || _die "--arena-branch requires a value"
      _ARENA_BRANCH="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,35p' "${BASH_SOURCE[0]}"
      if [[ "${_MANIPARENA_INSTALL_SOURCED}" -eq 1 ]]; then return 0; fi
      exit 0
      ;;
    *)
      _die "unknown argument: $1"
      ;;
  esac
done

_prefer_https_github() {
  # Arena's nested .gitmodules may use git@github.com; rewrite for HTTPS clones.
  local repo
  for repo in "${_ROOT}" "${_ARENA_DIR}"; do
    [[ -e "${repo}/.git" || -f "${repo}/.git" ]] || continue
    git -C "${repo}" config --local --unset-all "url.https://github.com/.insteadOf" 2>/dev/null || true
    git -C "${repo}" config --local --add "url.https://github.com/.insteadOf" "git@github.com:"
    git -C "${repo}" config --local --add "url.https://github.com/.insteadOf" "ssh://git@github.com/"
  done
}

_ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  _log "uv not found; installing via https://astral.sh/uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || _die "uv install failed; install manually: https://docs.astral.sh/uv/"
}

_preflight() {
  _log "preflight: OS/GPU/tools"
  if ! command -v git >/dev/null 2>&1; then
    _die "git is required"
  fi
  if ! command -v curl >/dev/null 2>&1; then
    _die "curl is required"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -3 || true
  else
    _log "WARNING: nvidia-smi not found; Isaac Sim 6 requires an NVIDIA GPU + driver"
  fi
  if command -v nvcc >/dev/null 2>&1; then
    _log "nvcc: $(nvcc --version | tail -1)"
  else
    _log "NOTE: nvcc not required for wheel-based Isaac Sim; CUDA 12.8 toolkit recommended"
  fi
  if [[ -n "${ISAACSIM_PATH:-}" ]]; then
    if [[ -d "${ISAACSIM_PATH}" ]]; then
      _log "ISAACSIM_PATH=${ISAACSIM_PATH} (recorded; primary install still uses Arena uv Sim wheels)"
    else
      _log "WARNING: ISAACSIM_PATH=${ISAACSIM_PATH} does not exist; ignoring"
    fi
  fi
}

_prepare_submodules() {
  _prefer_https_github
  mkdir -p "${_ROOT}/3rd"

  if [[ ! -e "${_ARENA_DIR}/.git" && ! -f "${_ARENA_DIR}/.git" ]]; then
    if [[ -f "${_ROOT}/.gitmodules" ]] && git -C "${_ROOT}" config -f .gitmodules --get-regexp path 2>/dev/null | grep -q "${_ARENA_REL}"; then
      _log "initializing submodule ${_ARENA_REL} (${_ARENA_BRANCH})"
      git -C "${_ROOT}" submodule sync -- "${_ARENA_REL}"
      git -C "${_ROOT}" submodule update --init --depth 1 -- "${_ARENA_REL}"
    else
      _log "adding submodule ${_ARENA_REL} -> ${_ARENA_URL} (${_ARENA_BRANCH})"
      git -C "${_ROOT}" submodule add -b "${_ARENA_BRANCH}" --depth 1 "${_ARENA_URL}" "${_ARENA_REL}"
    fi
  else
    _log "updating Arena checkout to origin/${_ARENA_BRANCH}"
    git -C "${_ARENA_DIR}" fetch --depth 1 origin "${_ARENA_BRANCH}"
    git -C "${_ARENA_DIR}" checkout -B "${_ARENA_BRANCH}" "origin/${_ARENA_BRANCH}"
  fi

  # Only Isaac Lab is required for ManipArena; skip Isaac-GR00T and friends.
  _log "initializing nested Isaac Lab submodule (skip GR00T)"
  git -C "${_ARENA_DIR}" submodule sync -- "submodules/IsaacLab"
  git -C "${_ARENA_DIR}" submodule update --init --depth 1 -- "submodules/IsaacLab"

  git -C "${_LAB_DIR}" remote set-url origin "https://github.com/isaac-sim/IsaacLab.git"
  if [[ "${_LAB_BRANCH}" == "arena-pin" || "${_LAB_BRANCH}" == "pinned" ]]; then
    local lab_sha
    lab_sha="$(git -C "${_ARENA_DIR}" ls-tree HEAD submodules/IsaacLab | awk '{print $3}')"
    [[ -n "${lab_sha}" ]] || _die "Arena does not record a Lab submodule SHA"
    _log "pinning Isaac Lab to Arena submodule SHA ${lab_sha}"
    # Shallow submodule may only contain the previous tip; fetch the pinned SHA.
    if ! git -C "${_LAB_DIR}" cat-file -e "${lab_sha}^{commit}" 2>/dev/null; then
      git -C "${_LAB_DIR}" fetch --depth 1 origin "${lab_sha}"
    fi
    git -C "${_LAB_DIR}" checkout --detach "${lab_sha}"
  else
    _log "WARNING: overriding Lab pin to ${_LAB_BRANCH}; Arena imports may break"
    # Nested submodule checkouts are often shallow and lack origin/<branch>.
    git -C "${_LAB_DIR}" fetch --depth 1 origin "${_LAB_BRANCH}"
    git -C "${_LAB_DIR}" checkout -B "${_LAB_BRANCH}" FETCH_HEAD
  fi

  # Prior develop checkouts leave gitignored package trees (contrib/core) that
  # confuse debugging; remove them when pinning to Arena's Lab layout.
  rm -rf \
    "${_LAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/contrib" \
    "${_LAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/core"

  if [[ -f "${_LAB_DIR}/VERSION" ]]; then
    _log "Isaac Lab VERSION=$(tr -d '\n' <"${_LAB_DIR}/VERSION") @ $(git -C "${_LAB_DIR}" rev-parse --short HEAD)"
  fi

  _patch_arena_lean_imports
}

_patch_arena_lean_imports() {
  # Soften upstream Arena eager imports so EX001 teleop/nav does not require
  # the fat RL / multi-robot stack (rsl-rl, onnxruntime/G1, ...).
  local emb_init="${_ARENA_DIR}/isaaclab_arena/embodiments/__init__.py"
  local policy_init="${_ARENA_DIR}/isaaclab_arena/policy/__init__.py"
  local registries="${_ARENA_DIR}/isaaclab_arena/assets/registries.py"
  [[ -f "${emb_init}" ]] || _die "missing ${emb_init}"
  [[ -f "${policy_init}" ]] || _die "missing ${policy_init}"
  [[ -f "${registries}" ]] || _die "missing ${registries}"

  _log "patching Arena embodiments/__init__.py for lean EX001 imports"
  cat >"${emb_init}" <<'EOF'
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
#
# ManipArena lean-stack override (applied by install.sh):
# Upstream eagerly imports agibot/droid/franka/g1/... here. That forces
# optional deps (e.g. onnxruntime for G1 WBC) even when only EX001 is used.
# Explicit submodule imports remain available, e.g.:
#   from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
EOF

  _log "patching Arena policy/__init__.py to skip optional RSL-RL policy"
  cat >"${policy_init}" <<'EOF'
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
#
# ManipArena lean-stack override (applied by install.sh):
# Keep zero/replay policies; RSL-RL is optional and needs rsl-rl-lib + isaaclab_rl.
from .replay_action_policy import *
from .zero_action_policy import *

try:
    from .rsl_rl_action_policy import *
except ImportError:
    pass
EOF

  # ensure_assets_registered() imports every Arena library; soft-fail optionals.
  if ! grep -q "ManipArena lean-stack override" "${registries}"; then
    _log "patching Arena assets/registries.py ensure_assets_registered for soft imports"
    python3 - "${registries}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
old = '''def ensure_assets_registered():
    """Ensure all assets are registered. Call this before accessing the registry."""
    global _assets_registered, _registration_in_progress
    if _assets_registered or _registration_in_progress:
        return
    _registration_in_progress = True
    try:
        # Import modules to trigger asset registration via decorators
        import isaaclab_arena.assets.background_library  # noqa: F401
        import isaaclab_arena.assets.device_library  # noqa: F401
        import isaaclab_arena.assets.hdr_image_library  # noqa: F401
        import isaaclab_arena.assets.object_library  # noqa: F401
        import isaaclab_arena.assets.retargeter_library  # noqa: F401
        import isaaclab_arena.assets.simready_object_library  # noqa: F401
        import isaaclab_arena.embodiments  # noqa: F401
        import isaaclab_arena.policy  # noqa: F401
        import isaaclab_arena.relations.relations  # noqa: F401
        import isaaclab_arena.tasks.task_library  # noqa: F401

        _assets_registered = True
    finally:
        _registration_in_progress = False
'''
new = '''def ensure_assets_registered():
    """Ensure all assets are registered. Call this before accessing the registry."""
    # ManipArena lean-stack override (applied by install.sh): soft-import optional
    # Arena libraries so missing RL/robot extras do not block EX001 teleop/nav.
    global _assets_registered, _registration_in_progress
    if _assets_registered or _registration_in_progress:
        return
    _registration_in_progress = True
    try:
        import importlib

        for _mod in (
            "isaaclab_arena.assets.background_library",
            "isaaclab_arena.assets.device_library",
            "isaaclab_arena.assets.hdr_image_library",
            "isaaclab_arena.assets.object_library",
            "isaaclab_arena.assets.retargeter_library",
            "isaaclab_arena.assets.simready_object_library",
            "isaaclab_arena.embodiments",
            "isaaclab_arena.policy",
            "isaaclab_arena.relations.relations",
            "isaaclab_arena.tasks.task_library",
        ):
            try:
                importlib.import_module(_mod)
            except Exception:
                # Optional for ManipArena lean stack; maniparena_sim registers its own assets.
                pass

        _assets_registered = True
    finally:
        _registration_in_progress = False
'''
if old not in text:
    raise SystemExit(f"ensure_assets_registered block not found in {path}")
path.write_text(text.replace(old, new, 1))
PY
  fi
}

_venv_python() {
  echo "${_ARENA_DIR}/.venv/bin/python"
}

_ensure_venv() {
  _ensure_uv
  if [[ ! -x "$(_venv_python)" ]]; then
    _log "creating Arena venv (Python 3.12) at ${_ARENA_REL}/.venv"
    # Seed pip so `pip install` works after activate (uv venvs omit pip by default).
    uv venv --python 3.12 --seed "${_ARENA_DIR}/.venv"
  elif [[ ! -x "${_ARENA_DIR}/.venv/bin/pip" ]]; then
    _log "seeding pip into existing Arena venv"
    uv pip install --python "$(_venv_python)" pip setuptools wheel
  fi
}

_pip() {
  # Isaac Sim wheels live on pypi.nvidia.com and pull prerelease pins
  # (e.g. tinyobjloader==2.0.0rc13). Plain PyPI resolution skips them.
  uv pip install --python "$(_venv_python)" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow \
    "$@"
}

_install_lean_stack() {
  # Minimal packages required to import ArenaEnvBuilder + run maniparena teleop/nav.
  # Explicitly avoids Arena's fat `isaaclab-from-source` group (mimic/rl/openpi/...).
  local lab_pkgs=(
    "isaaclab_assets"
    "isaaclab_physx"
    "isaaclab_newton"
    "isaaclab_ov"
    "isaaclab_ovphysx"
    "isaaclab_tasks"
    "isaaclab_teleop"
  )
  local pkg

  _ensure_venv
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
  export ACCEPT_EULA="${ACCEPT_EULA:-Y}"

  # Install Sim wheels first (explicit). Editable isaaclab[isaacsim] alone often
  # fails to resolve extras against NVIDIA's index when run via bare uv pip.
  _log "lean stack: isaacsim[all,extscache]==6.0.1.0 from pypi.nvidia.com"
  _pip "isaacsim[all,extscache]==6.0.1.0"

  # Do NOT use isaaclab[isaacsim] here: Lab's coverage pin conflicts with
  # isaacsim-kernel==6.0.1.0 (coverage==7.4.4). Sim wheels are already installed.
  _log "lean stack: editable isaaclab (Lab core, no [isaacsim] extra)"
  _pip -e "${_LAB_SRC}/isaaclab"
  # Lab may pull coverage!=7.4.4; restore the Sim wheel pin.
  _pip "coverage==7.4.4"

  for pkg in "${lab_pkgs[@]}"; do
    _log "lean stack: editable ${pkg}"
    _pip -e "${_LAB_SRC}/${pkg}"
  done

  # Lab 6 --viz kit requires isaaclab_visualizers.kit (KitVisualizerCfg).
  # Extra [kit] has no extra wheels; skip newton/rerun/viser backends.
  _log "lean stack: editable isaaclab_visualizers[kit] (for --viz kit)"
  _pip -e "${_LAB_SRC}/isaaclab_visualizers[kit]"
  _pip "coverage==7.4.4"

  _log "lean stack: common runtime wheels used by Lab/Arena/ManipArena"
  # hydra-core: required by isaaclab_tasks.utils.parse_env_cfg (ArenaEnvBuilder).
  # Not declared in isaaclab_tasks setup.py; pin stable (avoid --prerelease 1.4.dev).
  _pip \
    "numpy>=2" \
    "gymnasium>=1.2.0" \
    "h5py>=3.15.0" \
    prettytable \
    tqdm \
    packaging \
    psutil \
    "trimesh" \
    "einops" \
    "matplotlib>=3.10.3" \
    "hydra-core>=1.3.2,<1.4" \
    "omegaconf>=2.3,<2.4" \
    "pyyaml>=6.0" \
    "filelock" \
    "pillow" \
    "pyarrow>=14"

  _log "lean stack: isaaclab_arena editable (--no-deps; skip openai/sbi/onnxruntime/pytest/...)"
  _pip -e "${_ARENA_DIR}" --no-deps
  # Re-apply lean import patches after editable install (source tree is used as-is).
  _patch_arena_lean_imports
  # Thin deps actually needed for maniparena teleop (vuer) + Arena config types.
  # Note: current vuer release has no `all` extra.
  _pip \
    typing_extensions \
    "pydantic>=2.0" \
    vuer \
    scipy \
    decorator \
    "pandas==2.2.3"
}

_install_full_arena_stack() {
  _ensure_uv
  _log "FULL Arena stack: uv sync isaaclab-from-source (heavy; includes mimic/rl/tasks extras)"
  _log "openpi still excluded; this path is only for debugging upstream Arena installs"
  (
    cd "${_ARENA_DIR}"
    export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
    export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
    uv sync --no-default-groups --group isaaclab-from-source --no-group openpi
  )
}

_install_sim_stack() {
  if [[ "${_SKIP_STACK}" -eq 1 ]]; then
    _log "skipping Python stack install (--skip-stack)"
    return 0
  fi
  if [[ "${_FULL_ARENA_STACK}" -eq 1 ]]; then
    _install_full_arena_stack
  else
    _install_lean_stack
  fi
}

_install_maniparena() {
  local venv_py
  venv_py="$(_venv_python)"
  if [[ ! -x "${venv_py}" ]]; then
    if [[ "${_SKIP_STACK}" -eq 1 ]]; then
      _log "Arena venv missing; skip package install (run without --skip-stack to create it)"
      return 0
    fi
    _die "Arena venv missing at ${_ARENA_DIR}/.venv"
  fi
  _ensure_uv
  _log "installing maniparena_sim editable into Arena venv"
  (
    cd "${_ROOT}"
    uv pip install --python "${venv_py}" -e .
  )
}

_verify() {
  if [[ "${_SKIP_VERIFY}" -eq 1 ]]; then
    return 0
  fi
  local venv_py="${_ARENA_DIR}/.venv/bin/python"
  [[ -x "${venv_py}" ]] || return 0
  _log "verifying imports"
  "${venv_py}" - <<'PY'
import importlib
mods = [
    "isaacsim",
    "isaaclab",
    "isaaclab_arena",
    "maniparena_sim",
    "hydra",
    "omegaconf",
    "isaaclab_tasks.utils.parse_cfg",
]
failed = []
for name in mods:
    try:
        importlib.import_module(name)
        print(f"  OK  {name}")
    except Exception as exc:  # noqa: BLE001 — report all import failures
        failed.append((name, exc))
        print(f"  FAIL {name}: {exc}")
if failed:
    raise SystemExit(1)
print("import smoke test passed")
PY
}

_print_next_steps() {
  cat <<EOF

========================================================================
ManipArena-Sim install finished.

Activate the environment:

  source ${_ARENA_REL}/.venv/bin/activate
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

Optional smoke (needs display or headless AppLauncher setup):

  python -c "import isaaclab_arena, maniparena_sim; print('ok')"

Collect example (Lab/Sim 6.0 needs --viz kit for the Kit UI):

  python scripts/collect.py --robot ex001 --task fruits_to_basket \\
    --control-mode keyboard --config configs/collect/keyboard.yaml --viz kit

Details: docs/install.md
========================================================================
EOF
}

_install_body() {
  # Runs with set -e; keep this in a subshell when the script is sourced.
  set -euo pipefail
  cd "${_ROOT}"
  _preflight
  _prepare_submodules
  _install_sim_stack
  _install_maniparena
  if ! _verify; then
    _log "WARNING: import verification failed; activate venv and inspect errors"
  fi
  _print_next_steps
}

_activate_venv_in_current_shell() {
  if [[ ! -f "${_ARENA_DIR}/.venv/bin/activate" ]]; then
    return 0
  fi
  _log "sourcing Arena venv into current shell"
  # shellcheck disable=SC1091
  source "${_ARENA_DIR}/.venv/bin/activate"
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
  export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
}

if [[ "${_MANIPARENA_INSTALL_SOURCED}" -eq 1 ]]; then
  # Subshell isolates set -e / exit from the interactive parent shell.
  if ( _install_body "$@" ); then
    _activate_venv_in_current_shell
  else
    _status=$?
    _log "install failed (exit ${_status}); your shell is still open"
    return "${_status}"
  fi
else
  _install_body "$@"
fi
