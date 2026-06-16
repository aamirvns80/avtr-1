#!/usr/bin/env bash
#
# setup_cloud_gpu.sh — one-shot setup for AVTR-1 on a rented Linux NVIDIA GPU.
#
# Paste-and-go: copy this file to your GPU box (or curl it) and run:
#
#     bash setup_cloud_gpu.sh
#
# It is idempotent — safe to re-run; it skips steps that are already done.
#
# What it does:
#   1. Preflight: confirm Linux x86_64 + a working NVIDIA driver (nvidia-smi).
#   2. Install pixi (if missing).
#   3. Clone avtr-1 (if not already inside it).
#   4. Point AVTR1_LOCAL_STORAGE at a persistent path.
#   5. pixi install (resolve the CUDA env).
#   6. Download weights from HuggingFace (needs a HF token; see HF_TOKEN below).
#   7. Build the TensorRT engines (once per GPU).
#   8. Print the commands to run offline generation and the live demo.
#
# Optional environment variables (export before running, or edit below):
#   AVTR1_LOCAL_STORAGE   where weights + engines go   (default: ./artifacts under the repo)
#   HF_TOKEN              HuggingFace token for non-interactive login
#                         (otherwise you'll be prompted by `hf auth login`)
#   REPO_DIR             where to clone               (default: $HOME/avtr-1)
#   SKIP_DOWNLOAD=1      skip the weight download step
#   SKIP_BUILD=1         skip the TRT engine build step
#
set -euo pipefail

# ---------------------------------------------------------------- helpers ----
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

REPO_URL="https://github.com/avaturn-live/avtr-1.git"
REPO_DIR="${REPO_DIR:-$HOME/avtr-1}"

# ---------------------------------------------------------- 1. preflight ----
log "Preflight checks"

[ "$(uname -s)" = "Linux" ]  || die "AVTR-1 requires Linux. Detected: $(uname -s)."
[ "$(uname -m)" = "x86_64" ] || die "AVTR-1 requires x86_64. Detected: $(uname -m)."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. This box has no usable NVIDIA driver — AVTR-1 needs
       an NVIDIA GPU with CUDA 12.x. Pick a GPU instance/image with the driver
       installed (driver >= 550 recommended)."
fi

echo "GPU(s) detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || die "nvidia-smi failed to query the GPU. Fix the host driver first."

# ---------------------------------------------------------- 2. install pixi ----
if ! command -v pixi >/dev/null 2>&1; then
  log "Installing pixi"
  curl -fsSL https://pixi.sh/install.sh | sh
  # Make pixi available in THIS shell session.
  export PATH="$HOME/.pixi/bin:$PATH"
else
  log "pixi already installed: $(pixi --version)"
fi
command -v pixi >/dev/null 2>&1 || die "pixi still not on PATH; open a new shell and re-run."

# ---------------------------------------------------------- 3. clone repo ----
if [ -f "pixi.toml" ] && grep -q "avtr1-live-talkinghead" pixi.toml 2>/dev/null; then
  log "Already inside the avtr-1 checkout: $(pwd)"
  REPO_DIR="$(pwd)"
elif [ -d "$REPO_DIR/.git" ]; then
  log "Repo already cloned at $REPO_DIR"
  cd "$REPO_DIR"
else
  log "Cloning avtr-1 into $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

# ------------------------------------------------------ 4. storage path ----
export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-$REPO_DIR/artifacts}"
mkdir -p "$AVTR1_LOCAL_STORAGE"
log "Weights + engines will live in: $AVTR1_LOCAL_STORAGE"
warn "For cheaper restarts, point AVTR1_LOCAL_STORAGE at a PERSISTENT volume
      (e.g. /workspace/avtr1_storage) so you don't re-download on every reboot."

# ---------------------------------------------------------- 5. pixi install ----
log "Resolving the CUDA environment (pixi install) — first run pulls a few GB"
pixi install

# ---------------------------------------------------------- 6. weights ----
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  log "SKIP_DOWNLOAD=1 — skipping weight download"
else
  log "Downloading model weights from HuggingFace"
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "Using HF_TOKEN for non-interactive login."
    pixi run hf auth login --token "$HF_TOKEN" --add-to-git-credential || \
      warn "hf token login failed; the download step may prompt interactively."
  else
    warn "No HF_TOKEN set — 'pixi run download' will prompt you to paste a token.
          Create one (read scope) at https://huggingface.co/settings/tokens"
  fi
  pixi run download
fi

# ---------------------------------------------------------- 7. TRT engines ----
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  log "SKIP_BUILD=1 — skipping TRT engine build"
else
  log "Building TensorRT engines (once per GPU; takes several minutes)"
  pixi run build-trt-engines
fi

# ---------------------------------------------------------- 8. next steps ----
log "Setup complete."
cat <<EOF

You're ready. From inside $REPO_DIR (with AVTR1_LOCAL_STORAGE set):

  # --- OFFLINE: render an mp4 (works on any GPU) -------------------------
  pixi run generate_offline --speech example/speaker_1.ogg --bg plain_white
  pixi run generate_offline --duration 10 --bg plain_white        # idle motion
  # copy the result back to your laptop:
  #   scp <user>@<this-host>:$REPO_DIR/demo_output.mp4 .

  # --- LIVE: interactive WebRTC demo (needs a real-time GPU) ------------
  pixi run interactive-demo            # serves http://127.0.0.1:8081/
  # view it from your laptop via an SSH tunnel:
  #   ssh -L 8081:127.0.0.1:8081 <user>@<this-host>
  # then open http://127.0.0.1:8081/  and paste an OpenAI/Cartesia key in the UI.
  # If video won't connect, set Cloudflare TURN before launching:
  #   export CLOUDFLARE_TURN_KEY_ID=...   CLOUDFLARE_TURN_KEY_TOKEN=...

Avatar ids:      ls "\$AVTR1_LOCAL_STORAGE"/v1/avatars_artifacts/reference_frames/
Full reference:  see RUNNING_ON_CLOUD_GPU.md
EOF
