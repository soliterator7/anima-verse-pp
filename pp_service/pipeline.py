"""Post-processing pipeline with selectable method + configurable fallback.

Methods:
  internal  - local InsightFace swap (+ optional GFPGAN enhance). Self-contained.
  comfyui   - ReActor single-identity swap via a ComfyUI server.
  multiswap - multi-identity swap via a ComfyUI server.

Selection logic per request:
  1. Build an ordered candidate list: [requested-or-default] + config.FALLBACK_CHAIN.
  2. Walk it, skipping methods that are disabled (config.method_enabled) or not
     ready (comfy methods need their COMFY_*_URL). Try the first usable one.
  3. If a tried method errors or produces nothing, continue down the chain.
  4. internal is the usual safety net at the end of the chain.

The caller (main project) sends an image + the canonical reference faces and,
optionally, a method/options override. What happens is decided here; the main
project carries no swap logic.
"""
import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from . import config, engine_comfy, engine_enhance, engine_swap

logger = logging.getLogger("pp_service.pipeline")


@dataclass
class PostprocessResult:
    image_bytes: Optional[bytes]
    changed: bool
    method: str = "internal"
    swapped_faces: int = 0
    enhanced: bool = False
    notes: List[str] = field(default_factory=list)


def method_usable(method: str) -> bool:
    """Enabled AND ready. internal is ready whenever enabled."""
    if not config.method_enabled(method):
        return False
    if method == "internal":
        return True
    return engine_comfy.is_ready(method)


def _candidate_order(requested: Optional[str]) -> List[str]:
    first = (requested or config.DEFAULT_METHOD or "internal").lower()
    order = [first] + [m for m in config.FALLBACK_CHAIN if m != first]
    # de-dupe while preserving order
    seen, out = set(), []
    for m in order:
        if m in config.VALID_METHODS and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _internal(target_bytes, reference_bytes, do_swap, do_enhance, notes):
    img = Image.open(io.BytesIO(target_bytes)).convert("RGB")
    result_bgr = np.array(img)[:, :, ::-1]
    changed = False
    swapped = 0

    if do_swap and reference_bytes:
        target_faces = engine_swap.detect_faces(result_bgr)
        if not target_faces:
            notes.append("no face detected in target; swap skipped")
        else:
            source_faces = []
            for idx, rb in enumerate(reference_bytes):
                sf = engine_swap.first_face(rb)
                if sf is None:
                    notes.append(f"reference {idx} has no detectable face; skipped")
                else:
                    source_faces.append(sf)

            if not source_faces:
                notes.append("no usable reference faces; swap skipped")
            elif len(source_faces) == 1:
                for tf in target_faces:
                    result_bgr = engine_swap.swap_one(result_bgr, tf, source_faces[0])
                    swapped += 1
            else:
                pairs = min(len(target_faces), len(source_faces))
                for i in range(pairs):
                    result_bgr = engine_swap.swap_one(result_bgr, target_faces[i], source_faces[i])
                    swapped += 1
                if len(target_faces) != len(source_faces):
                    notes.append(
                        f"face/reference count mismatch (targets={len(target_faces)}, "
                        f"refs={len(source_faces)}); swapped {pairs} by position"
                    )
            changed = changed or swapped > 0

    enhanced = False
    if do_enhance and config.ENHANCE_MODEL:
        before = result_bgr
        result_bgr = engine_enhance.enhance_bgr(result_bgr)  # no-op if unavailable
        enhanced = result_bgr is not before
        changed = changed or enhanced

    if not changed:
        return PostprocessResult(None, False, "internal", 0, False, notes)

    out = Image.fromarray(result_bgr[:, :, ::-1])
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return PostprocessResult(buf.getvalue(), True, "internal", swapped, enhanced, notes)


def _comfy(method, target_bytes, reference_bytes, notes):
    if not reference_bytes:
        notes.append(f"{method}: no reference faces supplied")
        return None
    if method == "multiswap":
        out = engine_comfy.multiswap(target_bytes, reference_bytes)
    else:
        out = engine_comfy.faceswap(target_bytes, reference_bytes)
    if not out:
        notes.append(f"{method}: produced no image")
        return None
    return PostprocessResult(out, True, method, len(reference_bytes), False, notes)


def postprocess(
    target_bytes: bytes,
    reference_bytes: List[bytes],
    *,
    method: Optional[str] = None,
    do_swap: Optional[bool] = None,
    do_enhance: Optional[bool] = None,
) -> PostprocessResult:
    notes: List[str] = []
    do_swap = config.DEFAULT_DO_SWAP if do_swap is None else do_swap
    do_enhance = config.DEFAULT_DO_ENHANCE if do_enhance is None else do_enhance

    candidates = _candidate_order(method)
    usable = [m for m in candidates if method_usable(m)]
    if not usable:
        notes.append(f"no usable method among {candidates}; check PP_ENABLE_* / COMFY_*_URL")
        return PostprocessResult(None, False, candidates[0] if candidates else "internal",
                                 0, False, notes)
    if usable[0] != candidates[0]:
        notes.append(f"requested '{candidates[0]}' not usable; using fallback order {usable}")

    for m in usable:
        try:
            if m == "internal":
                res = _internal(target_bytes, reference_bytes, do_swap, do_enhance, notes)
            else:
                res = _comfy(m, target_bytes, reference_bytes, notes)
        except Exception as e:  # noqa: BLE001
            logger.exception("method %s failed", m)
            notes.append(f"{m}: error {e}")
            res = None
        if res is not None and res.changed:
            return res
        notes.append(f"{m}: no result, trying next" if m != usable[-1] else f"{m}: no result")

    return PostprocessResult(None, False, usable[0], 0, False, notes)
