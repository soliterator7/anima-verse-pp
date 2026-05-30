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
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from . import config, engine_comfy, engine_enhance, engine_swap, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pp_service")

app = FastAPI(title="anima-versa-pp")


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
                        "url_set": bool(config.COMFY_FACESWAP_URL),
                        "ref_slots": engine_comfy.slot_count("comfyui")},
            "multiswap": {**method_info("multiswap"),
                          "url_set": bool(config.COMFY_MULTISWAP_URL),
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


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
