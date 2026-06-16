# AVTR-1 Architecture

This document describes how AVTR-1 is put together: the two packages, how data
flows through them, and the design decisions worth knowing before you touch the
code. It complements the [README](README.md) (which covers install and usage)
and the per-module docstrings (which cover specifics).

> AVTR-1 is a flow-matching autoregressive talking-head model. Given a portrait
> image and two audio streams (the avatar's own speech + the audio it is
> "listening" to), it renders lip-synced, actively-listening video at 25 fps,
> real-time on a single NVIDIA GPU.

---

## 1. The two packages

| Package | Role | License |
| --- | --- | --- |
| [`src/avtr1_renderer/`](src/avtr1_renderer/) | **GPU inference engine.** Audio → motion → rendered frames. Stateless-per-request HTTP service. | PolyForm Noncommercial |
| [`src/avaturn_live_streamer/`](src/avaturn_live_streamer/) | **Live orchestration backend.** WebRTC transport, conversation engines (LLM/TTS), audio scheduling, frame timing. Calls the renderer over HTTP. | PolyForm Noncommercial |

`scripts/` (build / download / demo tooling) is under the more permissive AVTR-1
Community License. See [LICENSE.md](LICENSE.md) for the full component map.

The split is deliberate: the **renderer** is a pure, stateless-per-call GPU
service that knows nothing about conversations or transport; the **streamer**
holds all the async orchestration and treats the renderer as a black-box HTTP
endpoint. You can run the renderer on its own (offline generation, or behind a
load balancer for a GPU fleet) without ever touching the streamer.

---

## 2. The fundamental quantum: 5 frames / 200 ms

Almost every magic number in the codebase derives from one unit:

```
VIDEO_FPS                = 25
FRAMES_PER_RENDER_CHUNK  = 5
FRAME_DURATION           = 1/25 s = 40 ms
frame_len                = 640 samples  = 16000 / 25  (one frame of 16 kHz audio)
present window           = 5 frames = 200 ms = one render tick = one ODE batch = one HTTP request
chunk audio length       = (chunk_size + future_size) * frame_len + audio_shift
                         = (5 + 5) * 640 + 80 = 6480 samples
```

When you see `5`, `640`, `200ms`, `6480`, or `80`, they all trace back to here.
The whole system is pipelined around producing exactly **5 frames of motion per
step**.

---

## 3. Data flow, end to end (live session)

```
browser mic ─► WebRTC ─► streamer ─► conversation engine (OpenAI Realtime / Cartesia)
                                          │ SegmentGenerated events (on EventBus)
                                          ▼
                                   SpeechScheduler ×2  (present/future windowing → 5-frame quantum)
                                          │ RendererRequest
                                          ▼  POST /process-audio-v3   (state blob carried across calls)
                              ┌────────── avtr1_renderer (GPU) ──────────┐
                              │ HuBERT → encode → 5-step Euler ODE        │
                              │   (flow matching + baked CFG + AR(1) noise)│
                              │ → MotionFrame                             │
                              │ → stitch → warp → decode → putback → matte│
                              │ → pack (YUV420) → 1 batched H2D            │
                              └───────────────────────────────────────────┘
                                          │ state blob first, then 5 frames
                                          ▼
                                   VideoFrameGenerated events (timestamped to 25 fps grid)
                                          ▼
                              WebRTC video/audio tracks ─► browser (lip-synced playback)
```

**Offline** (`scripts/generate_offline.py`) is the same renderer core without the
streamer: slice an audio file into overlapping chunks, call
`pipeline.process_chunk` in a loop threading `state` forward, write frames to
ffmpeg.

The two architectural spines are:

1. The **EventBus** — coordination *within* one session.
2. The autoregressive **`State` blob** — coherence *across* chunks.

---

## 4. The renderer (`avtr1_renderer`)

### 4.1 Pipeline (`pipeline.py`)

`Pipeline` is a thin coordinator wiring engines together. `from_artifacts()`
downloads weights from HuggingFace, loads backgrounds as CUDA tensors, and loads
each model **TRT-first with ONNX fallback** — except the AVTR1 decoder, which
*requires* TensorRT (run `scripts/build_avtr1_engines.py` first). `process_chunk`
is the per-chunk hot path: it validates audio length, generates motion, and
returns `(next_state, frame_iterator)`.

### 4.2 Motion generation (`avtr1_motion_generator.py`) — the model core

`AVTR1MotionGenerator.generate_chunk` turns 5 frames of audio into 5 frames of
motion, entirely on the GPU:

- **Flow matching.** The model learns a velocity field `v(x, t)`; inference
  integrates `dx/dt = v` from noise (`t=0`) to clean motion (`t=1`) with **5
  forward-Euler steps** (`x += dt * v`). The field is nearly straight, so a
  handful of steps suffice (vs. dozens for diffusion).
- **Two-stage TRT.** `encode` runs once per chunk → 5 attention-ready condition
  tensors; `decode` runs once per ODE step.
- **Classifier-free guidance baked into the engine.** The 4-pass CFG batch
  (past / self-audio / other-audio / kp) lives *inside* the decode engine. The
  runtime only passes scalar weights (`w_self`, `w_other`, `w_kp`), tunable per
  request (`RenderOptions.cfg_*`) without rebuilding.
- **AR(1) progressive noise.** Initial noise is correlated frame-to-frame *and*
  across chunk boundaries (`noise_shared` carry) so motion has no jitter at the
  5-frame seams. `ε` is drawn from a truncated normal for stability.
- **Autoregressive memory.** `State.past_cond` (75 frames of normalized motion)
  and `State.audio_features` (75 frames of HuBERT features) shift forward each
  call. The model has ~3 s of context but HuBERTs only a small window per chunk.
- The 42-dim output (3 so3 rotation + 39 lipsync expression) is **z-score
  normalized**; `_motion_to_frames` de-normalizes into a `MotionFrame(R, exp)`.

### 4.3 Rendering (`renderer.py` + `components/putback.py`)

Motion → pixels, per chunk:

1. **stitch** — `MotionFrame` → LivePortrait source/driving keypoints.
2. **warp (once on the full chunk)** — the most expensive op; shares work across
   the 5 frames. Animates the avatar's 3D appearance volume. Needs a custom
   grid-sample TRT plugin.
3. **decode (per frame, b=1)** — SPADE decoder → 512×512 RGB face crop.
   Deliberately per-frame: batched SPADE is *slower* (no cross-batch sharing).
4. **putback** — affine warp the crop back to frame coords (using a **cached
   inverse `M_grid`** so render time pays 2 kernels, not kornia's ~40-kernel LU
   per call), mask-blend, MODNet alpha matte, composite over background.

`render_chunk_streaming` yields each frame as soon as it's ready (lowest latency,
live); `render_chunk` batches the whole chunk (offline).

### 4.4 Pixel packing (`components/pixel_format.py` + `frame_sink.py`)

GPU-side RGB(A) → packed uint8, BT.601 limited-range YUV 4:2:0:

- **`yuv_i420`** (1.5 bpp, default) — head already composited over background.
- **`yuv_i420_stacked_alpha`** (3 bpp) — a second I420 carries the matte as luma,
  passed through a precomputed LUT so a stock H.264/VP9/AV1 encoder reproduces
  alpha byte-exact. This ships transparency through codecs that have no alpha.

`pack_frames` does **one batched H2D** per chunk (bandwidth win over per-frame
copies).

### 4.5 Avatar registration (`avatar_loader.py`)

Runs once per portrait (ONNX, latency non-critical). The cascade:

```
PNG → composite over grey-200 (if RGBA) → SCRFD face detect → largest bbox
   → landmark106 → crop-224 → landmark203 → crop-512 (rotated)
   → resize-256 → appearance extractor (f_s)  +  motion extractor (KPInfo)
   → pre-warp pasteback mask  +  precompute inverse affine (M_grid)
```

Produces the immutable `Avatar` (CUDA-resident) reused by every render. Matting
is **auto-detected by channel count**: RGBA portrait → MODNet recovers alpha
(swappable backgrounds); RGB portrait → background baked in (`no_matting=True`).

### 4.6 Intro motion (`intro_motion.py`)

A pre-recorded clip ("with smile" / "without smile", selected by avatar id) plays
at session start so the avatar isn't frozen before the first audio. It is
relative-retargeted to the avatar's pose and normalized into `past_cond` rows,
seeding the autoregressive memory with coherent motion.

### 4.7 Runtime abstraction (`runtime/`)

`InferenceEngine[InputT, OutputT]` is a `Protocol`: input/output are dataclasses
whose **field names match the engine's tensor names**, every field a CUDA tensor.

- `TRTEngine` — zero host↔device copies, runs on the caller's current torch CUDA
  stream (no `synchronize()`), supports `out=` buffer reuse. Validates the
  dataclass↔engine schema at load.
- `OnnxRTEngine` — same Protocol via ONNX Runtime IOBinding; the fallback and a
  way to validate ONNX exports before TRT compilation.

This Protocol is why `load_engine(find_engine_or_onnx(...))` swaps backends
transparently.

### 4.8 HTTP API (`api/app.py`)

`POST /process-audio-v3` — audio in as raw int16 PCM @ 16 kHz mono (4 uploads:
current/future × speech/listen). Heavy work runs on a worker thread feeding an
anyio stream. **The response body is the new state blob first** (length in
`X-State-Length-Bytes`), **then frames concatenated** — so the client splits
without buffering the whole response. Also `/avatars`, `/health`.

### 4.9 Fleet scaling (`api/load_balancing.py`)

Each renderer process heartbeats `POST /worker/keep_alive` to an external load
balancer every ~0.5–1 s (jittered, to avoid thundering herds) and sends
`POST /worker/kill` on shutdown for clean drain. Wired into the FastAPI
`lifespan`. Set `LOAD_BALANCER_URL=disabled` for single-instance mode.

---

## 5. The streamer (`avaturn_live_streamer`)

### 5.1 EventBus + worklets (`event_bus.py`, `runner.py`)

`run_stream` spawns each **worklet** as a task in an `asyncio.TaskGroup`, each
getting a clone of a shared `EventBus`. Worklets communicate only through typed
events — never direct calls — which is what keeps the layers decoupled.

The bus is **type-routed** (dispatch by `type(event)`) with three notable
mechanisms:

- **Ready-barrier.** Starts unready; each `clone()` increments and each `ready()`
  decrements a counter. `publish()` blocks until the count hits 0 — so no event
  is lost to a not-yet-subscribed worklet. A timeout turns a crashed worker into
  a clear error instead of a deadlock.
- **Backpressure.** `publish()` blocks the producer when a subscriber queue is
  full — throttles fast producers to slow consumers.
- **`publish_nowait(allow_pre_ready=True)`.** For sync SDK callbacks that may fire
  before subscriptions exist; buffers and replays in arrival order.

### 5.2 Rendering worklet (`worklets/rendering.py`)

The bridge to the renderer. Two cooperating loops:

- `_listen_bus` consumes segment/speech events and feeds two `SpeechScheduler`s
  (avatar speech + user "listen" speech).
- `_render_frames_loop` is clock-driven: each tick steps both schedulers, calls
  the renderer, and publishes one `VideoFrameGenerated` per frame with precise
  timestamps.

### 5.3 Speech scheduling (`speech/`)

`SpeechScheduler.do_step` carves a **present** window (committed frames) plus a
**future** lookahead (so lip motion anticipates upcoming phonemes) out of
asynchronously-arriving TTS audio. It pads with silence only at segment
boundaries (never mid-word — it blocks and waits instead), shifting event
timestamps to preserve A/V sync. `interrupt()` (barge-in) truncates to the future
window and reports exact played duration.

`SpeechBuffer` is the immutable audio value type: **`Fraction`-based exact
duration** (no float drift over long sessions), rich slice/concat algebra, and
stateless `QQ`-quality resampling (chosen for fewer chunk-boundary artifacts).
The 24 kHz (OpenAI/native) ↔ 16 kHz (renderer) bridge happens here.

### 5.4 The render clock (`clocks.py`)

`StreamClocks.now` is wall-time minus accumulated processing delay — a
*compensated* clock. The render loop schedules the next tick to **finish half a
present-duration before its frames are due**, giving the GPU lead time. When a
chunk overruns, the lateness folds into `_total_delay`, so the whole timeline
shifts rather than A/V desyncing.

### 5.5 WebRTC transport (`localrtc/`)

`LocalRTC` owns the `aiortc` peer connection and three queues. Outbound video
**drops the oldest frame when full** (latency beats completeness for live video);
inbound mic audio is resampled and queued.

ICE resolution (`ice.py`) prefers, in order: manual TURN → Cloudflare TURN
(mints short-lived per-session credentials; the long-lived token never reaches
the browser) → STUN-only. The hard-won detail is **relay-only SDP filtering**
(`_filter_sdp_to_relay_only`): when the streamer is behind a UDP-blocking
firewall, host/srflx candidates are useless and (with Cloudflare TURN + Firefox)
can tear down the whole allocation; stripping them keeps only the working relay
path. A `/probe-offer` endpoint lets the browser discover which path actually
wins.

### 5.6 Conversation engines (`conversation_engines/`)

Pluggable "brain + voice" behind a discriminated union:

- **OpenAI Realtime** — single model does STT + LLM + TTS; semantic VAD turn
  detection (barge-in friendly).
- **Cartesia** — agent-based.

Both mint **ephemeral tokens** server-side and return a worklet
`(EventBus, StreamClocks) -> Coroutine` that publishes the `SegmentGenerated`
events the rendering worklet consumes. Credentials are supplied per-session from
the browser; the real API key is used once to mint, then discarded.

### 5.7 Local demo (`local_stream_cli.py`)

A self-contained FastAPI app that runs the whole pipeline with `aiortc` instead
of Daily — serves a browser UI, handles `/offer` WebRTC negotiation, and runs one
session at a time via `_SessionSlot`.

---

## 6. State: the persistence boundary

`AVTR1State` (autoregressive memory: past motion + audio history + AR-noise
carry) lives **entirely on CUDA** so the per-chunk hot path never crosses the
host boundary. The *only* place it touches the host is
`state_to_safetensors` / `state_from_safetensors`, used so the HTTP API can carry
state across stateless requests. A state blob is meaningful only paired with the
avatar it was created for.

---

## 7. Building TensorRT engines (`scripts/build_*_engines.py`)

TRT engines are **compute-capability specific**, so they are built per-machine,
not shipped. `build_avtr1_engines.py`:

1. Loads the scripted checkpoint, exports `encode` + `decode` to ONNX in memory.
2. Parses ONNX → TRT engine. FP16 by default, but **pins LayerNorm-flavored ops
   to FP32** for numerical stability (detected by name and the
   `Pow → ReduceMean` pattern).
3. Writes a `*_normalizer.safetensors` sidecar (lifted off the scripted module)
   so the runtime needs only the engines + sidecar, never the eager checkpoint.

The decode engine's CFG weights are **engine inputs** of shape `(latent_dim,)` —
which is what makes per-request guidance tuning possible without a rebuild.

---

## 8. Gotchas worth knowing

- **`cfg_kp` default differs by entry point.** `RenderOptions.cfg_kp = 3.0`
  (direct `Pipeline` API) vs. `4.0` (HTTP endpoint default). Pass it explicitly
  if you need parity.
- **Matting regime is decided by PNG channel count.** An RGB portrait bakes its
  background in and effectively ignores `bg_id`; use RGBA for swappable
  backgrounds.
- **Requires NVIDIA GPU + CUDA 12.x + TensorRT 10.x on Linux.** The project
  manifest only declares `linux-64`; there is no native Windows/CPU path.

---

## 9. Where to start reading

| If you want to understand… | Read |
| --- | --- |
| The model math | `avtr1_motion_generator.py` |
| Motion → pixels | `renderer.py`, `components/putback.py` |
| The HTTP contract | `api/app.py` |
| Session orchestration | `worklets/rendering.py`, `event_bus.py` |
| Audio timing | `speech/speech_scheduler.py`, `clocks.py` |
| Live transport | `localrtc/peer.py`, `localrtc/ice.py` |
| Running it end to end | `scripts/generate_offline.py` (offline), `local_stream_cli.py` (live) |
