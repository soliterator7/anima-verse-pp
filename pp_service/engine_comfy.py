"""ComfyUI-backed swap engines: ReActor FaceSwap and MultiSwap.

Both load a workflow JSON, patch only the input_* titled nodes with uploaded
images (and the input_*_use Crystools switches / model overrides), submit to a
ComfyUI server and return the resulting image. The ComfyUI server must have the
ReActor custom node (faceswap) or the flux2 multiswap graph's nodes + models
installed.

faceswap : single identity -> faceswap_reactor_api.json
           input_target_image + input_reference_image_1, ReActorFaceSwap gender.
multiswap: up to N identities -> multiswap_flux2_{gguf,safetensors}_api.json
           (selected by model_format). input_target_image +
           input_reference_image_1..N, each gated by an input_reference_image_N_use
           Crystools switch (boolean only), optional UNet override
           (input_gguf / input_safetensors) + input_clip override.
"""
import copy
import json
import logging
import threading
from typing import List, Optional

from . import comfy_client, config

logger = logging.getLogger("pp_service.comfy_engine")

_wf_cache = {}
_cache_lock = threading.Lock()
# ComfyUI prompts on one server must be serialized like any single backend.
_run_lock = threading.Lock()


def _load_workflow(path: str) -> dict:
    with _cache_lock:
        if path not in _wf_cache:
            with open(path, "r") as f:
                _wf_cache[path] = json.load(f)
        return copy.deepcopy(_wf_cache[path])


def _run(base_url: str, workflow: dict, target_bytes: bytes, ref_bytes: List[bytes],
         *, gender: Optional[str] = None, extra_widgets=None) -> Optional[bytes]:
    """Upload target + refs, patch the workflow, submit, return result bytes.

    Reference slots are filled in order. Slots with a matching
    input_reference_image_N_use switch are toggled on for filled slots and off
    for unfilled ones (so the workflow uses exactly the refs we supplied).
    Returns None if the workflow has no reference slots or no usable refs.
    """
    ref_titles = comfy_client.reference_image_titles(workflow)
    if not ref_titles:
        logger.error("workflow has no input_reference_image* nodes")
        return None
    if not ref_bytes:
        logger.warning("no reference images supplied")
        return None

    client = comfy_client.ComfyClient(base_url, timeout=config.COMFY_TIMEOUT)

    with _run_lock:
        if config.COMFY_FREE_MEMORY_BEFORE_RUN:
            client.free_memory()

        tgt_name = client.upload_image("pp_target.png", target_bytes)
        if not comfy_client.set_image_input(workflow, "input_target_image", tgt_name):
            logger.error("workflow has no input_target_image node")
            return None

        n_filled = 0
        for i, title in enumerate(ref_titles):
            use_title = f"{title}_use"
            if i < len(ref_bytes):
                up = client.upload_image(f"pp_ref_{i + 1}.png", ref_bytes[i])
                comfy_client.set_image_input(workflow, title, up)
                comfy_client.set_switch(workflow, use_title, True)
                n_filled += 1
            else:
                # Unused slot: turn its switch off if present.
                comfy_client.set_switch(workflow, use_title, False)

        if len(ref_bytes) > len(ref_titles):
            logger.warning("workflow has %d ref slots but %d refs supplied; extras ignored",
                           len(ref_titles), len(ref_bytes))

        if gender is not None:
            comfy_client.set_reactor_gender(workflow, gender)

        for title, key, value in (extra_widgets or []):
            if value:
                comfy_client.set_widget_input(workflow, title, key, value)

        logger.info("comfy run: %d ref slot(s), %d filled", len(ref_titles), n_filled)
        result = client.run(workflow)

    if result is None:
        return None
    return _match_source_size(target_bytes, result)


def _match_source_size(source_bytes: bytes, result_bytes: bytes) -> bytes:
    """Resize the result back to the source image's dimensions if they differ.

    flux2 multiswap can emit a fixed resolution (e.g. 1024x1024) regardless of
    the input size; keep PP output the same size as what was handed in.
    """
    import io
    from PIL import Image

    try:
        src = Image.open(io.BytesIO(source_bytes))
        res = Image.open(io.BytesIO(result_bytes)).convert("RGB")
        if res.size == src.size:
            return result_bytes
        res = res.resize(src.size, Image.LANCZOS)
        buf = io.BytesIO()
        res.save(buf, format="PNG")
        logger.info("resized comfy result %s -> %s", res.size, src.size)
        return buf.getvalue()
    except Exception:
        logger.exception("resize-to-source failed; returning raw result")
        return result_bytes


def faceswap(target_bytes: bytes, ref_bytes: List[bytes], base_url: Optional[str] = None,
             gender: str = "no") -> Optional[bytes]:
    base_url = base_url or config.COMFY_FACESWAP_URL
    if not base_url:
        raise RuntimeError("COMFY_FACESWAP_URL not configured")
    workflow = _load_workflow(config.COMFY_FACESWAP_WORKFLOW)
    return _run(base_url, workflow, target_bytes, ref_bytes, gender=gender)


def multiswap(target_bytes: bytes, ref_bytes: List[bytes], base_url: Optional[str] = None,
              model_format: Optional[str] = None) -> Optional[bytes]:
    base_url = base_url or config.COMFY_MULTISWAP_URL
    if not base_url:
        raise RuntimeError("COMFY_MULTISWAP_URL not configured")
    path, model_title, model_key = config.multiswap_workflow(model_format or "")
    workflow = _load_workflow(path)
    extra = [
        (model_title, model_key, config.COMFY_MULTISWAP_UNET),
        ("input_clip", "clip_name", config.COMFY_MULTISWAP_CLIP),
    ]
    return _run(base_url, workflow, target_bytes, ref_bytes, extra_widgets=extra)


def slot_count(method: str) -> int:
    """Number of reference slots the method's workflow exposes (0 on error)."""
    try:
        if method == "multiswap":
            path, _, _ = config.multiswap_workflow()
        elif method == "comfyui":
            path = config.COMFY_FACESWAP_WORKFLOW
        else:
            return 0
        return len(comfy_client.reference_image_titles(_load_workflow(path)))
    except Exception:
        return 0


def is_ready(method: str) -> bool:
    """A comfy method is ready when enabled AND its server URL is configured."""
    if method == "comfyui":
        return config.method_enabled("comfyui") and bool(config.COMFY_FACESWAP_URL)
    if method == "multiswap":
        return config.method_enabled("multiswap") and bool(config.COMFY_MULTISWAP_URL)
    return False


def reachable(method: str, timeout: float = 4.0) -> Optional[bool]:
    """Ping the method's ComfyUI server. None if no URL set, else True/False."""
    url = config.comfy_url(method)
    if not url:
        return None
    return comfy_client.ComfyClient(url, timeout=config.COMFY_TIMEOUT).health(timeout=timeout)
