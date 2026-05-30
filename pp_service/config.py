"""Configuration for the post-processing service.

All settings come from environment variables so the service can be deployed
independently of the main project. Model paths point at ONNX files under
./models (moved here from the main project); large binaries are gitignored.

Env var names keep the FACE_SERVICE_* / FACE_ENHANCE_* prefixes from the
original in-project face_service so existing setups keep working.

Method selection
----------------
Three post-processing methods exist: ``internal`` (local InsightFace swap +
GFPGAN enhance), ``comfyui`` (ReActor single-identity swap on a ComfyUI server)
and ``multiswap`` (multi-identity swap on a ComfyUI server).

Each can be enabled/disabled independently (``PP_ENABLE_*``) and an ordered
fallback chain decides what to try when the requested/default method is disabled
or fails (``PP_FALLBACK_CHAIN``). A method is only usable if it is both enabled
AND ready (internal: always; comfy methods: their ``COMFY_*_URL`` is set).
"""
import os


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _here(*parts: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


# --- Service ---------------------------------------------------------------
PORT = int(os.environ.get("PP_PORT") or os.environ.get("FACE_SERVICE_PORT") or "8005")
HOST = os.environ.get("PP_HOST", "0.0.0.0")

VALID_METHODS = ("internal", "comfyui", "multiswap")

# Method requested when the caller does not specify one.
DEFAULT_METHOD = (os.environ.get("PP_DEFAULT_METHOD", "internal").strip().lower() or "internal")

# Per-method enable switches. Internal defaults on; comfy methods default on too
# but are only *ready* once their URL is configured (see is_ready in pipeline).
ENABLE_INTERNAL = _flag("PP_ENABLE_INTERNAL", True)
ENABLE_COMFYUI = _flag("PP_ENABLE_COMFYUI", True)
ENABLE_MULTISWAP = _flag("PP_ENABLE_MULTISWAP", True)


def method_enabled(method: str) -> bool:
    return {
        "internal": ENABLE_INTERNAL,
        "comfyui": ENABLE_COMFYUI,
        "multiswap": ENABLE_MULTISWAP,
    }.get(method, False)


def _parse_chain(raw: str):
    out = []
    for part in raw.replace(";", ",").split(","):
        m = part.strip().lower()
        if m in VALID_METHODS and m not in out:
            out.append(m)
    return out


# Ordered fallback chain tried (after the requested method) when a method is
# disabled, not ready, or errors. Default ends at internal as the safety net.
FALLBACK_CHAIN = _parse_chain(os.environ.get("PP_FALLBACK_CHAIN", "internal")) or ["internal"]

# --- Models (now bundled under ./models) -----------------------------------
MODELS_DIR = os.environ.get("FACE_SERVICE_MODELS_DIR") or _here("models")


def _resolve_model(env_name: str, *candidates: str):
    """Return the env override, else the first existing candidate under MODELS_DIR."""
    val = os.environ.get(env_name)
    if val:
        return val
    for c in candidates:
        p = os.path.join(MODELS_DIR, c)
        if os.path.exists(p):
            return p
    return None


SWAP_MODEL = _resolve_model("FACE_SERVICE_MODEL_PATH", "reswapper_256.onnx", "inswapper_128.onnx") \
    or os.environ.get("FACESWAP_MODEL_PATH")
DET_SIZE = int(os.environ.get("FACE_SERVICE_DET_SIZE") or os.environ.get("FACESWAP_DET_SIZE") or "640")
OMP_THREADS = os.environ.get("FACE_SERVICE_OMP_NUM_THREADS") or os.environ.get("FACESWAP_OMP_NUM_THREADS")
DEBUG = _flag("FACE_SERVICE_DEBUG", False) or _flag("FACESWAP_DEBUG", False)

# --- Enhancement -----------------------------------------------------------
ENHANCE_MODEL = _resolve_model("FACE_ENHANCE_MODEL_PATH", "GFPGANv1.4.onnx", "codeformer.onnx", "GPEN-BFR-512.onnx")
ENHANCE_ENABLED = _flag("FACE_ENHANCE_ENABLED", True)
ENHANCE_BLEND = float(os.environ.get("FACE_ENHANCE_BLEND", "1.0"))
ENHANCE_COLOR_CORRECTION = _flag("FACE_ENHANCE_COLOR_CORRECTION", True)
ENHANCE_SHARPEN = _flag("FACE_ENHANCE_SHARPEN", True)
ENHANCE_SHARPEN_STRENGTH = float(os.environ.get("FACE_ENHANCE_SHARPEN_STRENGTH", "0.5"))
ENHANCE_CODEFORMER_WEIGHT = float(os.environ.get("FACE_ENHANCE_CODEFORMER_WEIGHT", "0.7"))

# --- Post-processing defaults (internal method) ----------------------------
DEFAULT_DO_SWAP = _flag("PP_DEFAULT_SWAP", True)
DEFAULT_DO_ENHANCE = _flag("PP_DEFAULT_ENHANCE", ENHANCE_ENABLED)

# --- ComfyUI workflows (ReActor FaceSwap + MultiSwap) ----------------------
# These methods submit a workflow JSON to a ComfyUI server. The server must have
# the ReActor custom node and its models installed in ITS OWN model directories —
# they are not the ./models files above.
WORKFLOWS_DIR = os.environ.get("PP_WORKFLOWS_DIR") or _here("workflows")

# ReActor single-identity face swap.
COMFY_FACESWAP_URL = os.environ.get("COMFY_FACESWAP_URL", "").strip()
COMFY_FACESWAP_WORKFLOW = os.environ.get("COMFY_FACESWAP_WORKFLOW") \
    or os.path.join(WORKFLOWS_DIR, "faceswap_reactor_api.json")

# Multi-identity swap (the bundled flux2 workflows expose 2 reference slots).
COMFY_MULTISWAP_URL = os.environ.get("COMFY_MULTISWAP_URL", "").strip()
COMFY_MULTISWAP_USE_V2 = _flag("COMFY_MULTISWAP_USE_V2", False)
COMFY_MULTISWAP_WORKFLOW = os.environ.get("COMFY_MULTISWAP_WORKFLOW") \
    or os.path.join(WORKFLOWS_DIR, "multiswap_flux2_api.json")
COMFY_MULTISWAP_WORKFLOW_V2 = os.environ.get("COMFY_MULTISWAP_WORKFLOW_V2") \
    or os.path.join(WORKFLOWS_DIR, "multiswap_flux2_v2_api.json")
# Optional model overrides patched into input_model / input_clip titled loaders.
COMFY_MULTISWAP_UNET = os.environ.get("COMFY_MULTISWAP_UNET", "").strip()
COMFY_MULTISWAP_CLIP = os.environ.get("COMFY_MULTISWAP_CLIP", "").strip()

# Free ComfyUI VRAM before each run (POST /free). Mirrors main project behaviour.
COMFY_FREE_MEMORY_BEFORE_RUN = _flag("COMFY_FREE_MEMORY_BEFORE_RUN", True)

# Shared ComfyUI client settings.
COMFY_TIMEOUT = int(os.environ.get("COMFY_TIMEOUT", "300"))
