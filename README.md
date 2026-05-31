# anima-verse-pp

Standalone **image post-processing service** for the anima-verse project.

The main project does **no** post-processing itself and never sends face bytes.
After anima-verse saves an eligible image it sends this service a tiny
notification (*"image X is ready"* — an identifier, no pixels). This service
then **pulls** the image plus the reference faces of the people in it, runs the
post-processing (face swap / enhancement), and writes the result back. What it
does between pull and write-back is entirely its own concern.

This separation exists so the main project carries no face-swap-specific code,
models, or configuration.

> **⚠ FaceSwap — legal responsibility.** This service includes a face-swap
> capability (InsightFace inswapper/reswapper, or a ComfyUI ReActor/MultiSwap
> workflow). It can paste a real person's face onto a generated image. **In many
> jurisdictions (incl. Germany / EU) doing this with someone's face without their
> explicit consent violates personality / image rights and may be a criminal
> offence.** Use face swapping only with images of yourself, fictional
> characters, or with the documented consent of the depicted person. The author
> is not responsible for misuse.

## How it fits together (pull model)

```
anima-verse                         anima-verse-pp
-----------                         -------------
saves image + sidecar JSON
GET /trigger?path=…&category=…  ──►  202 accepted (returns immediately)
                                     ├─ pull scene image + reference faces
   ◄── reads them via                │     • filesystem  (storage_dir set), or
       filesystem or gallery API     │     • anima-verse HTTP gallery API
                                     ├─ run pipeline (swap / enhance)
POST /api/images?path=…  ◄────────── └─ write result back (X-API-Key)
   sets sidecar postprocessed=true
```

Two pull modes, chosen automatically by `anima_verse.storage_dir`:

- **Filesystem** (same machine) — `storage_dir` points at the active world's
  storage dir; images and reference faces are read straight from disk.
- **Remote** (other machine) — `storage_dir` empty; images are fetched over
  anima-verse's HTTP gallery API. The write-back always uses the HTTP API with
  `X-API-Key`.

## Methods

| Method | What it does | Needs |
|---|---|---|
| `internal` | local InsightFace swap + GFPGAN enhance | nothing (CPU, self-contained) |
| `comfyui` | ReActor single-identity swap on a ComfyUI server | ComfyUI + ReActor node |
| `multiswap` | multi-identity swap (flux2) on a ComfyUI server | ComfyUI + flux2 workflow |

`default_method` picks the method when a request doesn't name one. `fallback`
is an ordered list tried when the chosen method is disabled / unreachable /
errors — `internal` is the safety net (no ComfyUI required). Each method can be
enabled/disabled independently in `config.yaml`.

## Configuration

Copy the template and edit it; `config.yaml` is gitignored (it holds your local
URLs), the template is committed:

```bash
cp config.example.yaml config.yaml
$EDITOR config.yaml
```

Key sections (see `config.example.yaml` for the full annotated template):

- `port` / `host` — service bind (default `8005` / `0.0.0.0`).
- `default_method`, `fallback` — method selection.
- `internal` / `comfyui` / `multiswap` — per-method `enabled` + ComfyUI `url`.
- `enhance` — GFPGAN tuning (blend, color correction, sharpen).
- `comfy` — ComfyUI client timeout + free-memory-before-run.
- `anima_verse` — `base_url`, `api_key` (must match anima-verse `server.api_key`),
  and `storage_dir` (set = filesystem mode, empty = remote mode).

Environment variables override everything in `config.yaml` (`PP_*`, `COMFY_*`,
`FACE_*`); precedence is **env > config.yaml > default**.

## Models & workflows

Model binaries and workflow JSONs live in the repo dirs but the large binaries
are gitignored:

- `models/` — `reswapper_256.onnx` (or `inswapper_128.onnx`) for swap,
  `GFPGANv1.4.onnx` (or `codeformer.onnx` / `GPEN-BFR-512.onnx`) for enhance.
  Auto-resolved from `./models`; override with `FACE_SERVICE_MODEL_PATH` /
  `FACE_ENHANCE_MODEL_PATH`. The InsightFace detection model (`buffalo_l`) is
  downloaded automatically on first run.
- `workflows/` — `faceswap_reactor_api.json`, `multiswap_flux2_api.json`,
  `multiswap_flux2_v2_api.json` (used by the `comfyui` / `multiswap` methods).

On start the server scans `models/` and `workflows/`, pings each configured
ComfyUI url, and prints what it found and which methods are READY.

## HTTP endpoints

```
GET  /health        -> status, default_method, fallback_chain, and per-method
                       { enabled, ready, url_set, reachable, ref_slots }

GET  /trigger?path=<world-relative>&category=<cat>
                    -> 202 accepted; pulls + processes + writes back in the
                       background (the anima-verse hand-off; no image bytes)

POST /postprocess   (multipart: image, refs[], meta) -> processed image / 204
                       direct push for testing/debugging (bytes in, bytes out)
POST /swap          (multipart: target, source)      -> swapped image  (debug)
POST /enhance       (multipart: image)               -> enhanced image (debug)
POST /reset         -> note to restart to reload models
```

`/trigger` is the production entry point (pull model). `/postprocess`, `/swap`
and `/enhance` are kept for local testing without anima-verse.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit it
./run.sh                              # start (scans models/workflows on boot)
./stop.sh                             # stop (pidfile + stray instances)
./test.sh                             # local /postprocess test wrapper
```

## Status

Standalone and in use. anima-verse triggers this service over `/trigger` and
takes the result back via `/api/images`. All post-processing — internal swap +
enhance, ComfyUI ReActor, flux2 MultiSwap — lives here.
