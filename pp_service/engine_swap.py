"""Face-swapping engine using InsightFace.

Adapted from the original in-project face_service/face_swap.py. Adds helpers so
the multi-reference pipeline can detect faces and swap them individually,
reusing a single loaded FaceAnalysis app + swapper model.

Runs in this isolated process to avoid ONNX/protobuf conflicts.
"""
import io
import logging
import os
import threading
from typing import Optional

import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger("pp_service.swap")

_swapper = None
_face_app = None
_init_lock = threading.Lock()
_initialized = False
_alignment_patched = False


def _resolve_default_swapper() -> str:
    candidates = ["inswapper_128.onnx", "reswapper_256.onnx"]
    for c in candidates:
        p = os.path.join(config.MODELS_DIR, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No swap model found in {config.MODELS_DIR}")


def _patch_face_alignment_for_256() -> None:
    """Monkey-patch insightface face_align for non-128 swap models.

    ReSwapper-256 needs an offset correction in face alignment because
    insightface's estimate_norm is only correct for 112 and 128.
    offset = (128/32768) * image_size - 0.5  (from somanchiu/ReSwapper)
    """
    global _alignment_patched
    if _alignment_patched:
        return
    try:
        import insightface.utils.face_align as face_align_module

        _original_estimate_norm = face_align_module.estimate_norm

        def patched_estimate_norm(lmk, image_size=112, mode="arcface"):
            M = _original_estimate_norm(lmk, image_size, mode)
            if image_size not in (112, 128):
                offset = (128 / 32768) * image_size - 0.5
                M[0, 2] += offset
                M[1, 2] += offset
            return M

        face_align_module.estimate_norm = patched_estimate_norm
        _alignment_patched = True
        logger.info("Face-alignment patch for 256px enabled")
    except Exception:
        logger.exception("Alignment patch failed")


def _load_swapper_model(model_path: str, providers):
    """Load the swapper model, routing by type.

    inswapper_128.onnx -> insightface.model_zoo.get_model() (standard path)
    reswapper_256.onnx -> direct INSwapper loading (bypasses ModelRouter, which
    would otherwise mis-instantiate a 256 model as ArcFaceONNX).
    """
    import onnxruntime

    basename = os.path.basename(model_path).lower()
    is_reswapper = "reswapper" in basename or "256" in basename

    if is_reswapper:
        from insightface.model_zoo.inswapper import INSwapper

        session = onnxruntime.InferenceSession(model_path, providers=providers)
        swapper = INSwapper(model_file=model_path, session=session)
        _patch_face_alignment_for_256()
        in_shape = session.get_inputs()[0].shape
        logger.info("ReSwapper loaded: input_size=%sx%s", in_shape[2], in_shape[3])
        return swapper

    import insightface
    return insightface.model_zoo.get_model(model_path, providers=providers)


def _do_init() -> None:
    global _swapper, _face_app
    import onnxruntime
    from insightface.app import FaceAnalysis

    if config.OMP_THREADS:
        os.environ.setdefault("OMP_NUM_THREADS", str(config.OMP_THREADS))
    os.environ.setdefault("OMP_PROC_BIND", "false")

    providers = onnxruntime.get_available_providers()
    logger.info("Available ONNX providers: %s", providers)

    det = min(config.DET_SIZE, 640)
    _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    _face_app.prepare(ctx_id=0, det_size=(det, det))

    swap_model_path = config.SWAP_MODEL or _resolve_default_swapper()
    logger.info("Loading swapper from: %s", swap_model_path)
    _swapper = _load_swapper_model(swap_model_path, providers)
    logger.info("Swapper loaded successfully")


def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        _do_init()
        _initialized = True


def is_loaded() -> bool:
    return _initialized and _swapper is not None and _face_app is not None


def get_face_app():
    ensure_initialized()
    return _face_app


# --- Low-level helpers (numpy BGR) ----------------------------------------

def detect_faces(img_bgr: np.ndarray) -> list:
    """Return detected faces, sorted left-to-right by bbox x0."""
    ensure_initialized()
    faces = _face_app.get(img_bgr)
    return sorted(faces, key=lambda f: float(f.bbox[0]))


def first_face(image_bytes: bytes) -> Optional[object]:
    """Detect the first/most prominent face in an encoded image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    bgr = np.array(img)[:, :, ::-1]
    faces = detect_faces(bgr)
    return faces[0] if faces else None


def swap_one(target_bgr: np.ndarray, target_face, source_face) -> np.ndarray:
    """Swap a single source face onto a single target face. Returns BGR array."""
    ensure_initialized()
    return _swapper.get(target_bgr, target_face, source_face, paste_back=True)


# --- High-level (encoded bytes) -------------------------------------------

def apply_face_swap(target_image_bytes: bytes, source_image_bytes: bytes) -> Optional[bytes]:
    """Swap the first source face onto ALL faces in target. Returns PNG bytes or None.

    Kept for backwards-compatible single-source behaviour (debug/legacy endpoint).
    """
    ensure_initialized()
    try:
        target_img = Image.open(io.BytesIO(target_image_bytes)).convert("RGB")
        source_img = Image.open(io.BytesIO(source_image_bytes)).convert("RGB")

        target_bgr = np.array(target_img)[:, :, ::-1]
        source_bgr = np.array(source_img)[:, :, ::-1]

        target_faces = detect_faces(target_bgr)
        source_faces = detect_faces(source_bgr)
        if not target_faces:
            logger.warning("No face in target")
            return None
        if not source_faces:
            logger.warning("No face in source")
            return None

        source_face = source_faces[0]
        result = target_bgr.copy()
        for tf in target_faces:
            result = swap_one(result, tf, source_face)

        out = Image.fromarray(result[:, :, ::-1])
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        logger.exception("Face swap error")
        return None
