# Running AVTR-1 on a Cloud GPU

AVTR-1 needs an **NVIDIA GPU with CUDA 12.x + TensorRT 10.x on Linux** — it will
not run on CPU, AMD/Intel GPUs, or Windows natively. If your local machine lacks
an NVIDIA GPU, rent one. This guide walks through the whole path: pick a box,
install, download weights, build engines, and run both the offline generator and
the live demo.

> The canonical install steps live in the [README](README.md); this guide adds
> the cloud-specific bits (GPU choice, SSH tunneling for the browser demo, TURN,
> and troubleshooting).

---

## 1. Pick a GPU

AVTR-1 generates motion in 5-frame chunks (200 ms of video per chunk at 25 fps).
A GPU is "real-time" if it renders a chunk in under 200 ms. From the README's
benchmarks:

| GPU | Latency / chunk | Real-time factor | Verdict |
| --- | --- | --- | --- |
| L40 | 84 ms | 2.4× | great |
| A100 | 91 ms | 2.2× | great |
| RTX 4060 Ti | 166 ms | 1.2× | fine |
| RTX 3070 | 181 ms | 1.1× | fine |
| L4 | 202 ms | 0.99× | borderline |
| RTX 3060 Ti | 206 ms | 0.97× | borderline |
| RTX 4060 | 232 ms | 0.86× | offline only |

**Recommendation:**
- **Live demo:** L40 / L4 / A10 / RTX 4090 / A100 (anything ≥ ~1.0× real-time).
- **Offline generation only:** anything works; real-time factor just affects how
  long the render takes, not whether it succeeds.
- **Architecture:** Ampere or newer (sm80+) is recommended — the TRT build script
  has an `--ampere-plus` fast path.
- **VRAM:** 24 GB is comfortable; 16 GB is usually enough at 720×1280.

**Where to rent** (any works — pick on price/availability):
- [RunPod](https://runpod.io) — cheap, "PyTorch 2.x / CUDA 12.x" templates, easy port exposure.
- [Lambda Cloud](https://lambdalabs.com) — A10/A100/H100, Ubuntu + CUDA preinstalled.
- [vast.ai](https://vast.ai) — cheapest spot pricing; filter for CUDA ≥ 12.4.
- AWS `g5.xlarge` (A10G) / `g6` (L4), GCP `g2` (L4), Azure `NV`-series.

Choose an image with a **recent NVIDIA driver** (CUDA 12.4+ runtime; driver
≥ 550). You do *not* need to install CUDA/TensorRT yourself — `pixi` brings the
toolchain; you only need a working NVIDIA driver on the host.

---

## 2. Verify the box

SSH in, then confirm the GPU and driver:

```bash
nvidia-smi          # should list your GPU + driver version (≥ 550 ideal)
nvcc --version || true   # not required — pixi provides cuda-nvcc
uname -a            # confirm Linux x86_64
```

If `nvidia-smi` fails, stop here — the driver is missing/broken and nothing below
will work. Fix the host driver (or pick a different image) first.

---

## 3. Install pixi + the project

```bash
# pixi (the project's package/env manager)
curl -fsSL https://pixi.sh/install.sh | sh
exec $SHELL          # reload PATH so `pixi` is found

git clone https://github.com/avaturn-live/avtr-1.git
cd avtr-1

# Optional: put weights + built engines somewhere with space (and on a
# persistent volume if your provider has one — saves re-downloading on restart).
export AVTR1_LOCAL_STORAGE=/workspace/avtr1_storage     # defaults to ./artifacts/

pixi install         # resolves the linux-64 / CUDA 12.8 environment from pixi.lock
```

`pixi install` reads `pixi.lock` and materializes the `renderer` env (torch cu128,
TensorRT, onnxruntime-gpu, cuda-nvcc, ffmpeg, …). First run pulls a few GB; give
it a few minutes.

---

## 4. Download model weights

```bash
pixi run download
```

This invokes `hf auth login` first (you'll paste a HuggingFace token — create one
at <https://huggingface.co/settings/tokens>, read scope is enough), then pulls:

- AVTR-1 weights from [`avaturn-live/avtr-1`](https://huggingface.co/avaturn-live/avtr-1)
- LivePortrait ONNX graphs from [`digital-avatar/ditto-talkinghead`](https://huggingface.co/digital-avatar/ditto-talkinghead)

Everything lands under `$AVTR1_LOCAL_STORAGE` (or `./artifacts/`).

---

## 5. Build the TensorRT engines (once per machine)

TRT engines are **compute-capability specific**, so they must be built on the
actual GPU you'll run on (not shipped). The AVTR1 decoder *requires* TRT; the rest
fall back to ONNX if you skip them, but building all is recommended.

```bash
# Everything at once:
pixi run build-trt-engines

# …or individually (same result):
pixi run build-trt-engines-avtr1       # required (decoder)
pixi run build-trt-engines-renderer
pixi run build-trt-engines-hubert
```

This step is CPU+GPU heavy and can take several minutes. Outputs go under
`$AVTR1_LOCAL_STORAGE`. Re-run only if you change GPUs.

> On Ampere+ you can pass through the script's `--ampere-plus` flag for a
> hardware-compatible build if you intend to move engines between sm80+ cards.

---

## 6. Offline generation (no browser, no GPU-realtime requirement)

The simplest way to confirm everything works end to end — renders an `.mp4`:

```bash
# Single speaker: avatar lip-syncs the audio.
pixi run generate_offline --speech example/speaker_1.ogg --bg plain_white

# Custom avatar + background:
pixi run generate_offline --speech example/speaker_1.ogg --avatar maria --bg minimal_office

# Two-speaker dialogue (avatar speaks --speech, reacts to --listen):
pixi run generate_offline --speech example/speaker_1.ogg --listen example/speaker_2.ogg \
  --avatar elena --out elena.mp4

# Silence / idle micro-motion for 10s:
pixi run generate_offline --duration 10 --bg plain_white
```

Available avatar ids = filenames (without `.png`) in
`$AVTR1_LOCAL_STORAGE/v1/avatars_artifacts/reference_frames/`.
Available background ids = filenames in the `backgrounds` artifact (e.g.
`plain_white`, `minimal_office`).

Copy the resulting `.mp4` back to your laptop to view:

```bash
# from your LOCAL machine:
scp user@<gpu-host>:/path/to/avtr-1/demo_output.mp4 .
```

---

## 7. Live interactive demo (browser + WebRTC + LLM voice)

The demo serves a localhost web UI and streams the avatar over WebRTC. It needs a
conversation engine (OpenAI Realtime **or** Cartesia) — credentials are entered
**in the browser** per session (never stored on the server).

### 7a. Start the server

```bash
pixi run interactive-demo
# serves on http://127.0.0.1:8081/ by default
```

### 7b. Reach the UI from your laptop

The server binds to `127.0.0.1` on the GPU box. Easiest + safest is an **SSH
tunnel** from your local machine:

```bash
# on your LOCAL machine:
ssh -L 8081:127.0.0.1:8081 user@<gpu-host>
# then open http://127.0.0.1:8081/ in your local browser
```

(Providers like RunPod can instead expose port 8081 directly via their proxy —
either works.)

### 7c. Configure in the browser

1. Pick a conversation engine and paste your API key:
   - **OpenAI Realtime** — an OpenAI API key with Realtime access.
   - **Cartesia** — a Cartesia API key + agent id.
2. Pick an avatar + background from the dropdowns (proxied from the renderer's
   `/avatars`).
3. Click **Start**, allow mic access, and talk to the avatar.

### 7d. TURN (only if the video won't connect)

WebRTC tries direct UDP first. On a cloud VM whose firewall/security-group blocks
inbound UDP (almost always the case on RunPod, AWS, etc.), the **media stream**
needs a **TURN relay** even though the page itself loads fine. Set up free
Cloudflare TURN — the full step-by-step is in **[§8](#8-cloudflare-turn-setup-live-demo-media-relay)** below.

> If you used an SSH tunnel (7b), direct UDP usually won't traverse it — configure
> TURN, or expose the port via your provider's proxy instead of tunneling.

---

## 8. Cloudflare TURN setup (live demo media relay)

Only needed for the **live demo** when video won't connect (it almost never will
on a cloud box without this). Free tier, no credit card. The page/signaling rides
your provider's HTTP proxy; **TURN carries only the audio/video media.**

### 8a. Create the TURN app

1. Sign in to **dash.cloudflare.com**.
2. Sidebar → **Realtime** → **TURN Server** (may appear under **Calls** /
   **Realtime Kit** on some accounts — same feature).
3. **Create TURN App** → name it (e.g. `avtr1-dev`) → **Create**.

### 8b. Copy the two credentials

On the app's detail page:

- **Turn Token ID** (a.k.a. *Turn Key ID*) — short identifier, UUID-without-dashes.
- **API Token** — long secret, **shown only once**. Copy it before leaving the page
  (lost tokens can't be re-viewed — create a new app or roll the token).

### 8c. Set them and launch

```bash
export CLOUDFLARE_TURN_KEY_ID="<Turn Token ID>"
export CLOUDFLARE_TURN_KEY_TOKEN="<API Token>"

pixi run interactive-demo --host 0.0.0.0 --port 8081
```

Each browser session mints a **fresh, short-lived** TURN credential via
Cloudflare's API — your long-lived API Token never leaves the server.

### 8d. Verify

1. **Server log**, on the first browser request:
   ```
   ice: using Cloudflare TURN
   ```
   If you instead see `ice: STUN-only (no TURN credentials configured)`, the env
   vars aren't set in the shell that launched the demo — re-export and relaunch.
2. **Browser** — the connectivity card under the controls should show
   **✓ relay via TURN** (the path that must pass on a cloud box).

### 8e. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Log says `STUN-only` | Env vars not in the launching shell. `export` both, then relaunch (not just in another session's `~/.bashrc`). |
| `Cloudflare TURN fetch failed` in log | Wrong/expired API Token, or Key ID and Token swapped. Re-copy both. |
| Relay check fails in browser | Most common cause: the **Key ID** was pasted into `CLOUDFLARE_TURN_KEY_TOKEN` instead of the full **API Token**. |
| TURN OK in log but no video | Check the OpenAI/Cartesia key in the UI — engine errors look like a dead stream. |

### 8f. Self-hosted alternative

Any standard TURN server works. Instead of the Cloudflare vars:

```bash
export TURN_URL="turn:your-coturn-host:3478"
export TURN_USERNAME="..."      # optional
export TURN_CREDENTIAL="..."    # optional
```

`resolve_ice_servers()` uses these verbatim (e.g. a self-hosted
[coturn](https://github.com/coturn/coturn)).

---

## 9. Running the renderer as a standalone service (optional)

For a multi-GPU fleet, the renderer is a plain FastAPI service:

```bash
pixi run python -m avtr1_renderer.api.app      # binds 0.0.0.0:8000
# POST /process-audio-v3  ·  GET /avatars  ·  GET /health
```

Set `LOAD_BALANCER_URL=<lb>` to have each worker heartbeat into a load balancer
(or `LOAD_BALANCER_URL=disabled` for single-instance). The streamer points at it
via its renderer config (`single` → instance URL, `load-balanced` → the LB).

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `pixi install` → `unsupported-platform … win-64` | You're on Windows/wrong OS. AVTR-1 is `linux-64` only — use a Linux GPU box. |
| `nvidia-smi: command not found` / driver errors | Host NVIDIA driver missing or too old. Fix the host image (driver ≥ 550). |
| `AVTR1 TRT engines not found` at runtime | You skipped step 5. Run `pixi run build-trt-engines-avtr1`. |
| CUDA OOM during build/run | Use a bigger-VRAM GPU, or lower resolution; close other GPU processes. |
| Engines error after moving to a new GPU | Engines are CC-specific — rebuild (step 5) on the new card. |
| `hf auth login` / 401 on download | Token missing/expired or no access to the gated repos. Re-run `pixi run download`. |
| Demo loads but no video | WebRTC can't traverse UDP — configure TURN (7d) or expose the port directly. |
| Demo "engine build failed" 400 | Bad/expired OpenAI/Cartesia key, or no Realtime access on the OpenAI key. |
| Slow / stuttering live video | GPU under ~1.0× real-time (see §1). Use a faster GPU or run offline. |

---

## 11. Cost-saving tips

- Put `$AVTR1_LOCAL_STORAGE` on a **persistent volume** so weights + engines
  survive instance restarts (downloads + engine builds are the slow parts).
- **Stop/terminate** the instance when idle — these GPUs bill by the minute/hour.
- For experimentation, **spot/community** instances (vast.ai, RunPod Community)
  are much cheaper; just expect occasional preemption.
- Build engines once on a persistent volume, then reuse across sessions on the
  same GPU type.

---

## TL;DR

```bash
# on a Linux NVIDIA-GPU box (Ampere+, CUDA driver ≥ 550):
curl -fsSL https://pixi.sh/install.sh | sh && exec $SHELL
git clone https://github.com/avaturn-live/avtr-1.git && cd avtr-1
export AVTR1_LOCAL_STORAGE=/workspace/avtr1_storage
pixi install
pixi run download                 # HF login + weights
pixi run build-trt-engines        # once per GPU
pixi run generate_offline --speech example/speaker_1.ogg --bg plain_white   # offline test
pixi run interactive-demo         # live demo on :8081  (SSH-tunnel to view)
```
