"""Minimal ComfyUI HTTP client (stdlib only).

Implements the upload -> queue_prompt -> poll history -> download flow used by
the ReActor FaceSwap and MultiSwap workflows, plus the title-based input_* node
patching convention (only nodes whose _meta.title starts with "input_" are
overwritten; everything else is the workflow author's design).

Ported from the main project's ComfyUIBackend client mechanics.
"""
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger("pp_service.comfy")


def _multipart(fields: Dict[str, str], files):
    """Build multipart/form-data. files: list of (name, filename, bytes)."""
    boundary = "----ppcomfy" + uuid.uuid4().hex
    crlf = b"\r\n"
    body = bytearray()
    for name, value in fields.items():
        body += b"--" + boundary.encode() + crlf
        body += f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf
        body += str(value).encode() + crlf
    for name, filename, content in files:
        body += b"--" + boundary.encode() + crlf
        body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode() + crlf
        body += b"Content-Type: image/png" + crlf + crlf
        body += content + crlf
    body += b"--" + boundary.encode() + b"--" + crlf
    return bytes(body), f"multipart/form-data; boundary={boundary}"


# --- node patching ---------------------------------------------------------

def find_nodes_by_title(workflow: dict, predicate) -> Dict[str, dict]:
    """Return {node_id: node} for nodes whose _meta.title satisfies predicate."""
    out = {}
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        title = (node.get("_meta") or {}).get("title", "")
        if title and predicate(title):
            out[nid] = node
    return out


def reference_image_titles(workflow: dict) -> List[str]:
    """Reference-image input slot titles present in the workflow, in slot order.

    ReActor faceswap uses 'input_reference_image_1'; multiswap uses
    'input_reference_image_1', 'input_reference_image_2', ... The '*_use'
    Crystools switch nodes are NOT slots — they toggle a slot on/off and are
    excluded here. Sorted numerically so slot order is stable.
    """
    titles = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        t = (node.get("_meta") or {}).get("title", "")
        if t.startswith("input_reference_image") and not t.endswith("_use"):
            titles.append(t)

    def _slot_no(title):
        tail = title.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    return sorted(set(titles), key=_slot_no)


def set_switch(workflow: dict, title: str, enabled: bool) -> bool:
    """Set ONLY the boolean on a Crystools 'Switch any' node (input_*_use).

    Per the workflow convention, on_true/on_false are the author's wiring and
    must never be touched — we flip only `boolean`.
    """
    if not title.startswith("input_"):
        raise ValueError(f"refusing to patch non-input node title: {title!r}")
    nodes = find_nodes_by_title(workflow, lambda t: t == title)
    if not nodes:
        return False
    for node in nodes.values():
        node.setdefault("inputs", {})["boolean"] = bool(enabled)
    return True


def set_reactor_gender(workflow: dict, gender: str) -> int:
    """Set detect_gender_input/source on every ReActorFaceSwap node. Returns count."""
    n = 0
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "ReActorFaceSwap":
            node.setdefault("inputs", {})["detect_gender_input"] = gender
            node["inputs"]["detect_gender_source"] = gender
            n += 1
    return n


def set_image_input(workflow: dict, title: str, server_filename: str) -> bool:
    """Set inputs.image on the LoadImage node with the given title. input_* only."""
    if not title.startswith("input_"):
        raise ValueError(f"refusing to patch non-input node title: {title!r}")
    nodes = find_nodes_by_title(workflow, lambda t: t == title)
    if not nodes:
        return False
    for node in nodes.values():
        node.setdefault("inputs", {})["image"] = server_filename
    return True


def set_widget_input(workflow: dict, title: str, key: str, value) -> bool:
    """Set an arbitrary inputs[key] on an input_* titled node (e.g. unet/clip name)."""
    if not title.startswith("input_"):
        raise ValueError(f"refusing to patch non-input node title: {title!r}")
    nodes = find_nodes_by_title(workflow, lambda t: t == title)
    if not nodes:
        return False
    for node in nodes.values():
        node.setdefault("inputs", {})[key] = value
    return True


# --- client ----------------------------------------------------------------

class ComfyClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _req(self, path: str, data: Optional[bytes] = None, headers=None, method=None):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def health(self) -> bool:
        try:
            with self._req("/system_stats") as r:
                return r.status == 200
        except Exception:
            return False

    def free_memory(self) -> None:
        """Best-effort POST /free to unload models before a run (never raises)."""
        try:
            payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
            with self._req("/free", data=payload,
                           headers={"Content-Type": "application/json"}, method="POST"):
                pass
        except Exception:
            logger.debug("ComfyUI /free failed (ignored)", exc_info=True)

    def upload_image(self, filename: str, content: bytes) -> str:
        """POST /upload/image, returns the server-side filename (subfolder-qualified)."""
        body, ctype = _multipart({"overwrite": "true", "type": "input"},
                                 [("image", filename, content)])
        with self._req("/upload/image", data=body, headers={"Content-Type": ctype}, method="POST") as r:
            info = json.loads(r.read().decode())
        name = info.get("name", filename)
        sub = info.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def queue_prompt(self, workflow: dict) -> str:
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        with self._req("/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST") as r:
            info = json.loads(r.read().decode())
        pid = info.get("prompt_id")
        if not pid:
            raise RuntimeError(f"ComfyUI /prompt returned no prompt_id: {info}")
        return pid

    def wait(self, prompt_id: str) -> dict:
        """Poll /history/{id} until the prompt finishes; returns its history entry."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                with self._req(f"/history/{prompt_id}") as r:
                    hist = json.loads(r.read().decode())
            except urllib.error.URLError:
                hist = {}
            if prompt_id in hist:
                return hist[prompt_id]
            time.sleep(1.0)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within {self.timeout}s")

    def download(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with self._req(f"/view?{q}") as r:
            return r.read()

    def first_output_image(self, history_entry: dict) -> Optional[bytes]:
        """Return bytes of the first SaveImage output in a finished history entry."""
        outputs = history_entry.get("outputs", {})
        for node_out in outputs.values():
            for img in node_out.get("images", []) or []:
                if img.get("type") == "temp":
                    continue
                return self.download(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
        # fall back to any image (incl. temp) if no permanent output found
        for node_out in outputs.values():
            for img in node_out.get("images", []) or []:
                return self.download(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
        return None

    def run(self, workflow: dict) -> Optional[bytes]:
        pid = self.queue_prompt(workflow)
        hist = self.wait(pid)
        return self.first_output_image(hist)
