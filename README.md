# anima-verse-pp

Standalone **image post-processing service** for the anima-verse project.

The main project does **no** post-processing itself. After it generates an
image, it hands the image plus the **canonical reference faces** of the persons
in the scene to this service over HTTP. What this service does with them — face
swap, enhancement, anything — is entirely its own concern. The main project
only sends *"here is an image and the identities in it"* and takes back the
result.

This separation exists so the main project carries no face-swap-specific code,
models, or configuration.

## Contract

```
GET  /health
  -> { available: true, swapper_loaded, enhancer_loaded, ... }

POST /postprocess          (multipart/form-data)
  image = <png>            the image to process
  refs  = <png> ...        reference face images (repeat the field once per person)
  meta  = <json string>    optional:
            {
              "references": [ {"slot":1,"character":"Kai","type":"female"}, ... ],
              "category": "scene" | "event" | "instagram" | ...,
              "options":  { "swap": true, "enhance": true }
            }
  -> 200 image/png         processed image (+ X-PP-* headers with a summary)
  -> 204                   nothing applicable changed
```

`refs` are ordered to match the face-bearing `references` entries (location /
background references are NOT sent). Matching of references to detected faces is
left-to-right positional (Phase 1 heuristic); single-reference requests swap
that identity onto every detected face.

Debug endpoints: `POST /swap` (target+source), `POST /enhance` (image).

## Models (not committed)

Large ONNX models live outside this repo and are referenced by env var:

| Env var | Purpose |
|---|---|
| `FACE_SERVICE_MODELS_DIR` | dir with `buffalo_l/` (detection) + swap model |
| `FACE_SERVICE_MODEL_PATH` | `inswapper_128.onnx` or `reswapper_256.onnx` |
| `FACE_ENHANCE_MODEL_PATH` | optional GFPGAN/CodeFormer/GPEN ONNX |
| `FACE_SERVICE_DET_SIZE` | detection size (default 640) |
| `PP_PORT` / `PP_HOST` | service bind (default 8005 / 0.0.0.0) |
| `PP_DEFAULT_SWAP` / `PP_DEFAULT_ENHANCE` | default actions when caller omits options |

The `FACE_*` env names are kept identical to the original in-project
`face_service` so existing model installations work unchanged.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# point the env vars at your models, then:
./run.sh
```

## Status

Phase 1: extracted from the main project's `face_service/` and made standalone
with a generic `/postprocess` hand-off. Once verified externally, the
face-swap code is removed from the main project (see
`development_instructions/plan-postprocessing-handoff.md` in anima-verse).
