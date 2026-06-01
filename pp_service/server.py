"""FastAPI server for the standalone post-processing service.

Primary endpoint:
    POST /postprocess  (multipart)
        image  = <png>             the image to process
        refs   = <png> ...         canonical reference faces (repeat field)
        meta   = <json string>     optional: { references:[{slot,character,type}],
                                               category,
                                               options:{method, swap, enhance} }
        -> 200 image/png           processed image
        -> 204                     no change (nothing applicable)

  options.method: "internal" (local InsightFace swap+enhance, default),
                  "comfyui"  (ReActor single-identity swap via ComfyUI),
                  "multiswap" (multi-identity swap via ComfyUI).

Support:
    GET  /health
    POST /reset
    POST /swap     (debug: target+source -> swapped, internal)
    POST /enhance  (debug: image -> enhanced, internal)
"""
import json
import logging
import os
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from . import config, engine_comfy, engine_enhance, engine_swap, handoff, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pp_service")

app = FastAPI(title="anima-verse-pp")


def startup_scan(ping: bool = True) -> None:
    """Print what the server found (config, models, workflows) and which methods
    are READY, pinging each configured ComfyUI url. Called from main()."""
    p = print
    p("=" * 64)
    p("anima-verse-pp  starting up")
    p("=" * 64)

    # config file
    if config.CONFIG_ERROR:
        p(f"[config]  ERROR reading {config.CONFIG_PATH}: {config.CONFIG_ERROR}")
        p("[config]  using defaults + environment only")
    elif config.CONFIG_LOADED:
        p(f"[config]  loaded {config.CONFIG_PATH}")
    else:
        p(f"[config]  no config.yaml at {config.CONFIG_PATH} (defaults + env)")

    # models
    def _m(path):
        if path and os.path.exists(path):
            return f"OK  {path}  ({os.path.getsize(path) // (1024*1024)} MB)"
        return f"MISSING  ({path or 'unset'})"
    p(f"[models]  dir: {config.MODELS_DIR}")
    p(f"[models]  swap   : {_m(config.SWAP_MODEL)}")
    p(f"[models]  enhance: {_m(config.ENHANCE_MODEL)}")

    # workflows
    def _w(path):
        return "OK" if path and os.path.exists(path) else "MISSING"
    p(f"[workflows] dir: {config.WORKFLOWS_DIR}")
    p(f"[workflows] faceswap (reactor): {_w(config.COMFY_FACESWAP_WORKFLOW)}")
    p(f"[workflows] multiswap gguf       : {_w(config.COMFY_MULTISWAP_WORKFLOW_GGUF)}")
    p(f"[workflows] multiswap safetensors: {_w(config.COMFY_MULTISWAP_WORKFLOW_SAFETENSORS)}")
    p(f"[workflows] multiswap format     : {config.COMFY_MULTISWAP_MODEL_FORMAT}")

    # methods
    p(f"[default] method: {config.DEFAULT_METHOD}   fallback: {config.FALLBACK_CHAIN}")
    for m in config.VALID_METHODS:
        enabled = config.method_enabled(m)
        if not enabled:
            p(f"[method] {m:9}: DISABLED")
            continue
        if m == "internal":
            ok = config.SWAP_MODEL is not None
            p(f"[method] {m:9}: {'READY' if ok else 'NOT READY (no swap model)'}")
            continue
        url = config.comfy_url(m)
        if not url:
            p(f"[method] {m:9}: enabled, but no url set -> NOT READY")
            continue
        slots = engine_comfy.slot_count(m)
        if ping:
            reach = engine_comfy.reachable(m)
            tag = "REACHABLE" if reach else "NOT REACHABLE"
            p(f"[method] {m:9}: url {url}  ({slots} ref slots)  ping -> {tag}")
        else:
            p(f"[method] {m:9}: url {url}  ({slots} ref slots)")

    p("=" * 64)
    p(f"serving on http://{config.HOST}:{config.PORT}")
    p("=" * 64)


@app.get("/health")
async def health():
    def method_info(m):
        return {
            "enabled": config.method_enabled(m),
            "ready": pipeline.method_usable(m),
        }

    return {
        "status": "ok",
        "available": True,
        "default_method": config.DEFAULT_METHOD,
        "fallback_chain": config.FALLBACK_CHAIN,
        "methods": {
            "internal": method_info("internal"),
            "comfyui": {**method_info("comfyui"),
                        "url": config.COMFY_FACESWAP_URL,
                        "url_set": bool(config.COMFY_FACESWAP_URL),
                        "reachable": engine_comfy.reachable("comfyui"),
                        "ref_slots": engine_comfy.slot_count("comfyui")},
            "multiswap": {**method_info("multiswap"),
                          "url": config.COMFY_MULTISWAP_URL,
                          "url_set": bool(config.COMFY_MULTISWAP_URL),
                          "reachable": engine_comfy.reachable("multiswap"),
                          "ref_slots": engine_comfy.slot_count("multiswap")},
        },
        "swapper_loaded": engine_swap.is_loaded(),
        "enhancer_loaded": engine_enhance.is_loaded(),
        "enhance_model_configured": bool(config.ENHANCE_MODEL),
    }


@app.post("/postprocess")
async def postprocess(
    image: UploadFile = File(...),
    refs: List[UploadFile] = File(default=[]),
    meta: Optional[str] = Form(default=None),
):
    try:
        target_bytes = await image.read()
        reference_bytes = [await r.read() for r in refs]

        method = do_swap = do_enhance = None
        if meta:
            try:
                opts = (json.loads(meta).get("options") or {})
                method = opts.get("method")
                if "swap" in opts:
                    do_swap = bool(opts["swap"])
                if "enhance" in opts:
                    do_enhance = bool(opts["enhance"])
            except (ValueError, AttributeError):
                logger.warning("ignoring unparsable meta payload")

        result = pipeline.postprocess(
            target_bytes, reference_bytes,
            method=method, do_swap=do_swap, do_enhance=do_enhance,
        )
        if not result.changed or result.image_bytes is None:
            return Response(status_code=204, headers={"X-PP-Notes": "; ".join(result.notes)[:500]})
        headers = {
            "X-PP-Method": result.method,
            "X-PP-Swapped-Faces": str(result.swapped_faces),
            "X-PP-Enhanced": "1" if result.enhanced else "0",
        }
        if result.notes:
            headers["X-PP-Notes"] = "; ".join(result.notes)[:500]
        return Response(content=result.image_bytes, media_type="image/png", headers=headers)
    except Exception as e:
        logger.exception("postprocess failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    return JSONResponse({"status": "reset", "note": "restart process to reload models"})


# --- debug / low-level endpoints ------------------------------------------

@app.post("/swap")
async def swap(target: UploadFile = File(...), source: UploadFile = File(...)):
    out = engine_swap.apply_face_swap(await target.read(), await source.read())
    if out is None:
        raise HTTPException(status_code=422, detail="Face swap failed")
    return Response(content=out, media_type="image/png")


@app.post("/enhance")
async def enhance(image: UploadFile = File(...)):
    out = engine_enhance.apply_face_enhance(await image.read())
    if out is None:
        raise HTTPException(status_code=422, detail="Enhancement unavailable or failed")
    return Response(content=out, media_type="image/png")


@app.get("/trigger")
async def trigger(path: str, category: str = "", background: BackgroundTasks = None):
    """anima-verse hand-off notification (pull model).

    anima-verse calls this after saving an eligible image, passing only an
    identifier (world-relative path) — no image bytes. We acknowledge
    immediately (202) and do the actual pull+process+write-back in the
    background so anima-verse never blocks.
    """
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    if background is not None:
        background.add_task(_run_handoff, path, category)
    else:  # pragma: no cover — BackgroundTasks is always injected by FastAPI
        _run_handoff(path, category)
    return JSONResponse(status_code=202, content={"status": "accepted", "path": path, "category": category})


def _run_handoff(path: str, category: str) -> None:
    try:
        result = handoff.process(path, category)
        logger.info("handoff result: %s", result)
    except Exception:
        logger.exception("handoff failed for %s", path)


def main():
    import uvicorn
    startup_scan(ping=True)
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
