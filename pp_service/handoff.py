"""anima-verse hand-off (pull model).

anima-verse calls GET /trigger?path=<world-relative>&category=<cat> after saving
an eligible image. This module then:
  1. reads the scene image (local filesystem, or anima-verse gallery URL),
  2. reads the sidecar JSON and resolves the reference faces it lists,
  3. runs the post-processing pipeline,
  4. writes the result back to anima-verse via POST /api/images (X-API-Key).

No image bytes ever travel in the trigger itself — this side pulls everything.
"""
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from . import config, pipeline

logger = logging.getLogger("pp_service.handoff")


# --- reading (local filesystem mode) ---------------------------------------

def _storage() -> Path:
    return Path(config.ANIMAVERSE_STORAGE_DIR).resolve()


def _read_scene_local(rel_path: str) -> Optional[bytes]:
    p = (_storage() / rel_path).resolve()
    if _storage() not in p.parents and _storage() != p:
        logger.error("scene path escapes storage: %s", rel_path)
        return None
    if not p.is_file():
        logger.error("scene image not found: %s", p)
        return None
    return p.read_bytes()


def _read_sidecar_local(rel_path: str) -> dict:
    side = (_storage() / rel_path).with_suffix(".json")
    if not side.is_file():
        return {}
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _resolve_reference_faces_local(meta: dict) -> List[bytes]:
    """Resolve the face reference images the sidecar recorded.

    The sidecar's `reference_images` maps slot -> filename of the actual
    reference used (e.g. a character profile image). We locate each filename
    under characters/*/images/. Location/background slots don't live there, so
    they are naturally filtered out — only real face refs remain.
    """
    refs: List[bytes] = []
    ref_map = meta.get("reference_images") or {}
    chars_dir = _storage() / "characters"
    for _slot, filename in ref_map.items():
        if not filename:
            continue
        # find <filename> under any character's images dir
        matches = list(chars_dir.glob(f"*/images/{filename}"))
        if matches and matches[0].is_file():
            try:
                refs.append(matches[0].read_bytes())
            except OSError:
                logger.warning("could not read reference face: %s", matches[0])
        else:
            logger.debug("reference not a character face (skipped): %s", filename)
    return refs


# --- reading (remote URL mode) ---------------------------------------------

def _http_get(url: str, timeout: int = 60) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.URLError as e:
        logger.error("GET failed %s: %s", url, e)
        return None


def _read_scene_url(rel_path: str) -> Optional[bytes]:
    # anima-verse serves character images at /characters/<name>/images/<file>.
    # The world-relative path already is characters/<name>/images/<file>, so we
    # can map it 1:1 onto the public endpoint.
    url = f"{config.ANIMAVERSE_BASE_URL}/{urllib.parse.quote(rel_path)}"
    return _http_get(url)


def _parse_char_and_file(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (character, filename) from characters/<char>/images/<file>."""
    parts = rel_path.split("/")
    if len(parts) >= 4 and parts[0] == "characters" and parts[2] == "images":
        return parts[1], parts[3]
    return None, None


def _http_get_json(url: str, timeout: int = 30) -> Optional[dict]:
    data = _http_get(url, timeout)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        logger.error("JSON parse failed %s: %s", url, e)
        return None


def _profile_image_url(character: str) -> bytes:
    base = config.ANIMAVERSE_BASE_URL
    return _http_get(f"{base}/characters/{urllib.parse.quote(character)}/images/profile")


def _resolve_reference_faces_url(rel_path: str) -> List[bytes]:
    """Resolve reference faces over HTTP (remote mode).

    1. GET /characters/<char>/images -> image_metadata[<file>].
    2. From the sidecar metadata read which persons are in the image
       (canonical.persons) and pull each one's profile image.

    Using canonical.persons (not reference_images filenames) keeps it robust:
    profile images get overwritten/renamed, but the character NAME is stable and
    /images/profile always returns the current profile. Location/background refs
    carry no person name, so they are naturally excluded.
    """
    char, filename = _parse_char_and_file(rel_path)
    if not char or not filename:
        logger.warning("url-mode: path is not a character image: %s", rel_path)
        return []

    base = config.ANIMAVERSE_BASE_URL
    listing = _http_get_json(f"{base}/characters/{urllib.parse.quote(char)}/images")
    if not listing:
        return []
    meta = (listing.get("image_metadata") or {}).get(filename) or {}

    # Persons in the image -> their profile faces, in order, de-duplicated.
    names: List[str] = []
    for person in ((meta.get("canonical") or {}).get("persons") or []):
        name = (person.get("name") or "").strip()
        if name and name not in names:
            names.append(name)

    refs: List[bytes] = []
    for name in names:
        face = _profile_image_url(name)
        if face:
            refs.append(face)
        else:
            logger.debug("url-mode: no profile image for %s", name)
    logger.info("url-mode: resolved %d reference face(s) from persons %s", len(refs), names)
    return refs


# --- writing back ----------------------------------------------------------

def _write_back(rel_path: str, image_bytes: bytes) -> Tuple[bool, str]:
    base = config.ANIMAVERSE_BASE_URL
    key = config.ANIMAVERSE_API_KEY
    if not key:
        return False, "anima_verse.api_key not configured"
    q = urllib.parse.urlencode({"path": rel_path})
    url = f"{base}/api/images?{q}"
    req = urllib.request.Request(url, data=image_bytes, method="POST")
    req.add_header("Content-Type", "image/png")
    req.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return (200 <= r.status < 300), f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except urllib.error.URLError as e:
        return False, f"connection error: {e}"


# --- orchestration ---------------------------------------------------------

def process(rel_path: str, category: str) -> dict:
    """Pull image + references, post-process, write result back. Returns a summary."""
    mode = config.animaverse_mode()
    logger.info("handoff: path=%s category=%s mode=%s", rel_path, category, mode)

    if mode == "local":
        scene = _read_scene_local(rel_path)
        meta = _read_sidecar_local(rel_path)
        refs = _resolve_reference_faces_local(meta)
    else:
        scene = _read_scene_url(rel_path)
        refs = _resolve_reference_faces_url(rel_path)

    if scene is None:
        return {"ok": False, "error": "could not read scene image", "path": rel_path}

    if not refs:
        # Without references we can still enhance, but a swap needs faces.
        logger.warning("handoff: no references resolved (only enhance possible)")

    result = pipeline.postprocess(scene, refs)
    if not result.changed or result.image_bytes is None:
        return {"ok": True, "changed": False, "path": rel_path,
                "method": result.method, "notes": result.notes}

    ok, detail = _write_back(rel_path, result.image_bytes)
    return {
        "ok": ok, "changed": True, "path": rel_path,
        "method": result.method, "swapped": result.swapped_faces,
        "enhanced": result.enhanced, "writeback": detail,
        "refs": len(refs),
    }
