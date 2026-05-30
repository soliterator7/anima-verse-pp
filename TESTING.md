# Testing the post-processing service

## 1. Install (GPU)

```bash
cd /home/dev/projekte/anima-versa-pp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # onnxruntime-gpu — needs matching CUDA/cuDNN
```

> If insightface needs to build, you may need `pip install cython numpy` first and
> system build tools. `onnxruntime-gpu` requires a CUDA runtime compatible with the
> wheel (same constraint as the existing face_service).

## 2. Point at the models (live outside this repo)

The existing models in the main project work as-is:

```bash
export FACE_SERVICE_MODELS_DIR=/home/dev/projekte/anima-verse/models
export FACE_SERVICE_MODEL_PATH=/home/dev/projekte/anima-verse/models/reswapper_256.onnx
export FACE_ENHANCE_MODEL_PATH=/home/dev/projekte/anima-verse/models/GFPGANv1.4.onnx
# optional: export FACE_SERVICE_DET_SIZE=640   PP_PORT=8005
```

`buffalo_l` (detection) is downloaded by insightface on first run into
`~/.insightface/` (or reuse the existing cache if present).

## 3. Start the service

```bash
./run.sh
# logs print model load + provider (CUDA vs CPU)
```

Check it loaded on GPU:

```bash
python test_postprocess.py --health
# expect swapper_loaded:false until first request; provider visible in run.sh logs
```

## 4. Run a real post-process

Pick a **scene image** (target, multiple/contextual faces) and the **profile
image(s)** of the person(s) in it as references. Profile images live under
`worlds/<world>/characters/<Char>/images/` (or the avatar/user images dir).

Single identity (swap that face onto every detected face in target):

```bash
python test_postprocess.py \
  /path/to/scene.png \
  /home/dev/projekte/anima-verse/worlds/demo/characters/<Char>/images/<profile>.png \
  -o /tmp/pp_out.png
```

Two identities (left-to-right positional match):

```bash
python test_postprocess.py scene.png charA_profile.png charB_profile.png -o /tmp/pp_out.png
```

Swap only / enhance only:

```bash
python test_postprocess.py scene.png ref.png --no-enhance -o /tmp/swap_only.png
python test_postprocess.py scene.png ref.png --no-swap   -o /tmp/enhance_only.png
```

## 5. Raw curl equivalents

```bash
# health
curl -s http://127.0.0.1:8005/health | python -m json.tool

# postprocess (one reference)
curl -s -X POST http://127.0.0.1:8005/postprocess \
  -F "image=@scene.png;type=image/png" \
  -F "refs=@charA_profile.png;type=image/png" \
  -F 'meta={"options":{"swap":true,"enhance":true}}' \
  -o /tmp/pp_out.png -D -
# -D - prints headers incl. X-PP-Swapped-Faces / X-PP-Enhanced / X-PP-Notes
```

## What to verify

- `/health` returns `available:true`; run.sh log shows `CUDAExecutionProvider`.
- A scene with a face produces an output where the face matches the reference.
- 2-person scene maps refs left-to-right (note any mismatch in `X-PP-Notes`).
- 204 response when target has no detectable face (nothing applicable).
- `reswapper_256.onnx` path triggers the 256px alignment patch (see log line).
