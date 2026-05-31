# Testing the post-processing service

## 1. Install (GPU)

```bash
cd /home/dev/projekte/anima-verse-pp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # onnxruntime-gpu — needs matching CUDA/cuDNN
```

> If insightface needs to build, you may need `pip install cython numpy` first and
> system build tools. `onnxruntime-gpu` requires a CUDA runtime compatible with the
> wheel. If insightface drags in a non-headless OpenCV (libxcb errors), fix with:
> `pip install --force-reinstall --no-deps opencv-python-headless`.

## 2. Models & config

Model binaries live in `./models` and are auto-resolved — no env vars needed:

```
models/reswapper_256.onnx   (or inswapper_128.onnx)   swap
models/GFPGANv1.4.onnx       (or codeformer / GPEN)    enhance
```

`buffalo_l` (detection) is downloaded by insightface on first run into
`~/.insightface/`. Override model paths with `FACE_SERVICE_MODEL_PATH` /
`FACE_ENHANCE_MODEL_PATH` only if you keep them elsewhere.

Everything else is in `config.yaml`:

```bash
cp config.example.yaml config.yaml   # then set ComfyUI urls / anima_verse api_key
```

## 3. Start the service

```bash
./run.sh
# startup scan prints models/workflows found, ComfyUI reachability,
# and which methods are READY (internal vs comfyui vs multiswap)
```

Check health (shows per-method enabled/ready/url/ref_slots):

```bash
./test.sh                 # = health
# or: python test_postprocess.py --health
```

## 4. Direct post-process (push, for local testing)

`./test.sh` is the quick wrapper. Pick a **scene image** (target) and the
**profile image(s)** of the people in it as references. Profile images live
under `worlds/<world>/characters/<Char>/images/`.

```bash
# default method (config default_method, with fallback):
./test.sh scene.png charA_profile.png

# force a method:
./test.sh scene.png charA_profile.png internal
./test.sh scene.png charA_profile.png comfyui
./test.sh scene.png charA_profile.png multiswap

# multiple identities (left-to-right positional match):
./test.sh scene.png charA_profile.png charB_profile.png
```

Lower-level via `test_postprocess.py` (swap/enhance toggles):

```bash
python test_postprocess.py scene.png ref.png --method internal --no-enhance -o /tmp/swap_only.png
python test_postprocess.py scene.png ref.png --method internal --no-swap   -o /tmp/enhance_only.png
```

Raw curl:

```bash
curl -s -X POST http://127.0.0.1:8005/postprocess \
  -F "image=@scene.png;type=image/png" \
  -F "refs=@charA_profile.png;type=image/png" \
  -F 'meta={"options":{"method":"internal","swap":true,"enhance":true}}' \
  -o /tmp/pp_out.png -D -
# -D - prints headers incl. X-PP-Method / X-PP-Swapped-Faces / X-PP-Enhanced / X-PP-Notes
```

## 5. Hand-off trigger (pull, the production path)

This is how anima-verse drives the service: it sends only a path, the service
pulls the image + references itself and writes the result back. Configure the
`anima_verse` section in `config.yaml` first (`base_url`, `api_key`, and
`storage_dir` for filesystem mode).

```bash
# world-relative path of an image anima-verse already saved:
curl -s "http://127.0.0.1:8005/trigger?path=characters/<Char>/images/<file>.png&category=scene"
# -> 202 accepted (immediately); pull+process+write-back runs in the background
```

Verify in the service log (`handoff result: …`) and check the image in
anima-verse — its sidecar JSON should now have `postprocessed: true` and the
gallery shows the **PP** badge.

## What to verify

- `/health` returns `available:true`; run.sh log shows `CUDAExecutionProvider`.
- A scene with a face produces an output where the face matches the reference.
- 2-person scene maps refs left-to-right (note any mismatch in `X-PP-Notes`).
- 204 response when the target has no detectable face (nothing applicable).
- `reswapper_256.onnx` path triggers the 256px alignment patch (see log line).
- Method fallback works: force an offline `comfyui`/`multiswap` and confirm it
  falls back to `internal` with a note.
- `/trigger` writes the result back and sets the sidecar `postprocessed` flag.
