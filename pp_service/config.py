"""Configuration for the post-processing service.

Settings come from ``config.yaml`` (edit that file, then run ./run.sh).
Precedence per value:  environment variable  >  config.yaml  >  built-in default.

So config.yaml is your normal place to set things; env vars (PP_*, COMFY_*,
FACE_*) override it for one-off runs / deployment.

Methods
-------
internal  - local InsightFace swap + GFPGAN enhance (no ComfyUI needed).
comfyui   - ReActor single-identity swap on a ComfyUI server.
multiswap - multi-identity swap (flux2) on a ComfyUI server.

Each can be enabled/disabled and an ordered ``fallback`` chain decides what to
try when the requested/default method is disabled, not ready, or errors. A
method is usable only if enabled AND ready (internal: always; comfy methods:
their url is set).
"""
import os

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _here(*parts: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


# --- load config.yaml ------------------------------------------------------
CONFIG_PATH = os.environ.get("PP_CONFIG") or _here("config.yaml")
_yaml = {}
CONFIG_LOADED = False
CONFIG_ERROR = None
if yaml is not None and os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH) as f:
            _yaml = yaml.safe_load(f) or {}
        CONFIG_LOADED = True
    except Exception as e:  # noqa: BLE001
        CONFIG_ERROR = str(e)


def _sec(name: str) -> dict:
    v = _yaml.get(name)
    return v if isinstance(v, dict) else {}


def _b(value, default: bool) -> bool:
    """Coerce a yaml/env scalar to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _flag(env_name: str, yaml_val, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return _b(yaml_val, default)


def _str(env_name: str, yaml_val, default: str = "") -> str:
    raw = os.environ.get(env_name)
    if raw is not None:
        return raw.strip()
    return str(yaml_val).strip() if yaml_val is not None else default


def _int(env_name: str, yaml_val, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw:
        return int(raw)
    return int(yaml_val) if yaml_val is not None else default


def _float(env_name: str, yaml_val, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw:
        return float(raw)
    return float(yaml_val) if yaml_val is not None else default


_internal = _sec("internal")
_comfyui = _sec("comfyui")
_multiswap = _sec("multiswap")
_enhance = _sec("enhance")
_comfy = _sec("comfy")

# --- Service ---------------------------------------------------------------
PORT = _int("PP_PORT", _yaml.get("port"), 8005)
if os.environ.get("FACE_SERVICE_PORT"):
    PORT = int(os.environ["FACE_SERVICE_PORT"])
HOST = _str("PP_HOST", _yaml.get("host"), "0.0.0.0")

VALID_METHODS = ("internal", "comfyui", "multiswap")

DEFAULT_METHOD = _str("PP_DEFAULT_METHOD", _yaml.get("default_method"), "internal").lower() or "internal"

ENABLE_INTERNAL = _flag("PP_ENABLE_INTERNAL", _internal.get("enabled"), True)
ENABLE_COMFYUI = _flag("PP_ENABLE_COMFYUI", _comfyui.get("enabled"), True)
ENABLE_MULTISWAP = _flag("PP_ENABLE_MULTISWAP", _multiswap.get("enabled"), True)


def method_enabled(method: str) -> bool:
    return {
        "internal": ENABLE_INTERNAL,
        "comfyui": ENABLE_COMFYUI,
        "multiswap": ENABLE_MULTISWAP,
    }.get(method, False)


def _parse_chain(value):
    """Accept a comma string (env) or a list (yaml); keep valid, ordered, unique."""
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = []
    out = []
    for p in parts:
        m = str(p).strip().lower()
        if m in VALID_METHODS and m not in out:
            out.append(m)
    return out


FALLBACK_CHAIN = (_parse_chain(os.environ.get("PP_FALLBACK_CHAIN"))
                  or _parse_chain(_yaml.get("fallback"))
                  or ["internal"])

# --- Models (bundled under ./models) ---------------------------------------
MODELS_DIR = os.environ.get("FACE_SERVICE_MODELS_DIR") or _here("models")


def _resolve_model(env_name: str, *candidates: str):
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
DET_SIZE = _int("FACE_SERVICE_DET_SIZE", None, 640)
OMP_THREADS = os.environ.get("FACE_SERVICE_OMP_NUM_THREADS") or os.environ.get("FACESWAP_OMP_NUM_THREADS")
DEBUG = _flag("FACE_SERVICE_DEBUG", None, False)

# --- Enhancement (internal) ------------------------------------------------
ENHANCE_MODEL = _resolve_model("FACE_ENHANCE_MODEL_PATH", "GFPGANv1.4.onnx", "codeformer.onnx", "GPEN-BFR-512.onnx")
ENHANCE_ENABLED = _flag("FACE_ENHANCE_ENABLED", _internal.get("enhance"), True)
ENHANCE_BLEND = _float("FACE_ENHANCE_BLEND", _enhance.get("blend"), 1.0)
ENHANCE_COLOR_CORRECTION = _flag("FACE_ENHANCE_COLOR_CORRECTION", _enhance.get("color_correction"), True)
ENHANCE_SHARPEN = _flag("FACE_ENHANCE_SHARPEN", _enhance.get("sharpen"), True)
ENHANCE_SHARPEN_STRENGTH = _float("FACE_ENHANCE_SHARPEN_STRENGTH", _enhance.get("sharpen_strength"), 0.5)
ENHANCE_CODEFORMER_WEIGHT = _float("FACE_ENHANCE_CODEFORMER_WEIGHT", _enhance.get("codeformer_weight"), 0.7)

# internal swap/enhance toggles (from the internal: section)
DEFAULT_DO_SWAP = _flag("PP_DEFAULT_SWAP", _internal.get("swap"), True)
DEFAULT_DO_ENHANCE = _flag("PP_DEFAULT_ENHANCE", _internal.get("enhance"), ENHANCE_ENABLED)

# --- ComfyUI workflows -----------------------------------------------------
WORKFLOWS_DIR = os.environ.get("PP_WORKFLOWS_DIR") or _here("workflows")

COMFY_FACESWAP_URL = _str("COMFY_FACESWAP_URL", _comfyui.get("url"), "")
COMFY_FACESWAP_WORKFLOW = os.environ.get("COMFY_FACESWAP_WORKFLOW") \
    or os.path.join(WORKFLOWS_DIR, "faceswap_reactor_api.json")

COMFY_MULTISWAP_URL = _str("COMFY_MULTISWAP_URL", _multiswap.get("url"), "")
# Model format selects the workflow: 'gguf' (LoaderGGUF / input_gguf) or
# 'safetensors' (UNETLoader / input_safetensors). The two graphs are otherwise
# identical (2 reference slots, shared CLIP/VAE).
COMFY_MULTISWAP_MODEL_FORMAT = _str(
    "COMFY_MULTISWAP_MODEL_FORMAT", _multiswap.get("model_format"), "gguf").lower()
COMFY_MULTISWAP_WORKFLOW_GGUF = os.environ.get("COMFY_MULTISWAP_WORKFLOW_GGUF") \
    or os.path.join(WORKFLOWS_DIR, "multiswap_flux2_gguf_api.json")
COMFY_MULTISWAP_WORKFLOW_SAFETENSORS = os.environ.get("COMFY_MULTISWAP_WORKFLOW_SAFETENSORS") \
    or os.path.join(WORKFLOWS_DIR, "multiswap_flux2_safetensors_api.json")
COMFY_MULTISWAP_UNET = _str("COMFY_MULTISWAP_UNET", _multiswap.get("unet"), "")
COMFY_MULTISWAP_CLIP = _str("COMFY_MULTISWAP_CLIP", _multiswap.get("clip"), "")

COMFY_FREE_MEMORY_BEFORE_RUN = _flag("COMFY_FREE_MEMORY_BEFORE_RUN", _comfy.get("free_memory_before_run"), True)
COMFY_TIMEOUT = _int("COMFY_TIMEOUT", _comfy.get("timeout"), 300)


def comfy_url(method: str) -> str:
    return COMFY_FACESWAP_URL if method == "comfyui" else (COMFY_MULTISWAP_URL if method == "multiswap" else "")


def multiswap_workflow(model_format: str = ""):
    """Resolve the multiswap workflow for a model format.

    Returns (workflow_path, model_node_title, model_key):
      gguf        -> LoaderGGUF node 'input_gguf'        (key 'gguf_name')
      safetensors -> UNETLoader node 'input_safetensors' (key 'unet_name')
    Unknown / empty format falls back to gguf.
    """
    fmt = (model_format or COMFY_MULTISWAP_MODEL_FORMAT or "gguf").lower()
    if fmt == "safetensors":
        return COMFY_MULTISWAP_WORKFLOW_SAFETENSORS, "input_safetensors", "unet_name"
    return COMFY_MULTISWAP_WORKFLOW_GGUF, "input_gguf", "gguf_name"


# --- anima-verse hand-off (pull model) -------------------------------------
_animaverse = _sec("anima_verse")
ANIMAVERSE_BASE_URL = _str("ANIMAVERSE_BASE_URL", _animaverse.get("base_url"), "http://127.0.0.1:8000").rstrip("/")
ANIMAVERSE_API_KEY = _str("ANIMAVERSE_API_KEY", _animaverse.get("api_key"), "")
ANIMAVERSE_STORAGE_DIR = _str("ANIMAVERSE_STORAGE_DIR", _animaverse.get("storage_dir"), "")


def animaverse_mode() -> str:
    """'local' if a storage dir is configured (filesystem read), else 'url'."""
    return "local" if ANIMAVERSE_STORAGE_DIR else "url"
