"""Face enhancement engine (GFPGAN / CodeFormer / GPEN ONNX).

Adapted from the original in-project face_service/face_enhance.py. Restores /
enhances faces after swapping.
"""
import io
import logging
import threading
from typing import Optional

import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger("pp_service.enhance")

_enhancer = None
_face_app = None
_init_lock = threading.Lock()
_initialized = False


def _do_init() -> None:
    global _enhancer, _face_app
    if not config.ENHANCE_ENABLED:
        logger.info("Face enhancement disabled")
        return
    if not config.ENHANCE_MODEL:
        logger.warning("FACE_ENHANCE_MODEL_PATH not set, enhancement unavailable")
        return
    import onnxruntime
    from insightface.app import FaceAnalysis

    providers = onnxruntime.get_available_providers()
    _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    _face_app.prepare(ctx_id=0, det_size=(512, 512))

    sess_options = onnxruntime.SessionOptions()
    _enhancer = onnxruntime.InferenceSession(config.ENHANCE_MODEL, sess_options, providers=providers)
    logger.info("Enhancer loaded from %s", config.ENHANCE_MODEL)


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
    return _initialized and _enhancer is not None


def _enhance_face_crop(face_img_rgb: np.ndarray) -> np.ndarray:
    """Run ONNX enhancement on a 512x512 aligned face crop (RGB)."""
    inp = face_img_rgb.astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5
    inp = np.transpose(inp, (2, 0, 1))[None, ...]
    ort_inputs = {_enhancer.get_inputs()[0].name: inp}
    try:
        if "codeformer" in (config.ENHANCE_MODEL or "").lower():
            w = np.array([config.ENHANCE_CODEFORMER_WEIGHT], dtype=np.float64)
            ort_inputs[_enhancer.get_inputs()[1].name] = w
    except Exception:
        pass
    output = _enhancer.run(None, ort_inputs)[0][0]
    output = np.clip((output * 0.5 + 0.5) * 255.0, 0, 255)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
    return output


def _paste_enhanced(img_bgr: np.ndarray, face) -> np.ndarray:
    """Align face to 512, enhance, paste back with blending. BGR in/out."""
    import cv2
    from skimage import transform as trans

    src = np.array([
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ], dtype=np.float32)
    landmarks = face.kps.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, src)
    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(img_bgr, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
    enhanced = _enhance_face_crop(aligned[:, :, ::-1])  # BGR->RGB for model
    enhanced = enhanced[:, :, ::-1]                       # back to BGR

    M_inv = cv2.invertAffineTransform(M)
    h, w = img_bgr.shape[:2]
    pasted = cv2.warpAffine(enhanced, M_inv, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = np.ones((512, 512), dtype=np.float32)
    mask = cv2.warpAffine(mask, M_inv, (w, h))
    mask = (mask > 0.5).astype(np.float32)[..., None]

    if config.ENHANCE_SHARPEN:
        blur = cv2.GaussianBlur(pasted, (0, 0), 3)
        pasted = cv2.addWeighted(
            pasted, 1 + config.ENHANCE_SHARPEN_STRENGTH,
            blur, -config.ENHANCE_SHARPEN_STRENGTH, 0,
        )

    blend = config.ENHANCE_BLEND
    if blend < 1.0:
        pasted = (pasted * blend + img_bgr * (1 - blend)).astype(np.uint8)

    result = (img_bgr * (1 - mask) + pasted * mask).astype(np.uint8)
    return result


def enhance_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Enhance all faces in a BGR array in place; returns BGR array.

    No-op (returns input) if the enhancer is unavailable.
    """
    ensure_initialized()
    if _enhancer is None:
        return img_bgr
    faces = _face_app.get(img_bgr)
    if not faces:
        return img_bgr
    result = img_bgr.copy()
    for face in faces:
        try:
            result = _paste_enhanced(result, face)
        except Exception:
            logger.exception("Enhance paste failed for one face")
    return result


def apply_face_enhance(image_bytes: bytes, face_app=None) -> Optional[bytes]:
    """Enhance all faces in an encoded image. Returns PNG bytes or None."""
    ensure_initialized()
    if _enhancer is None:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = np.array(img)[:, :, ::-1]
        out_bgr = enhance_bgr(bgr)
        out = Image.fromarray(out_bgr[:, :, ::-1])
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        logger.exception("Enhance error")
        return None
