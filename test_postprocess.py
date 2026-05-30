#!/usr/bin/env python3
"""Manual test client for the post-processing service.

Usage:
  # 1) start the service in another terminal:  ./run.sh
  # 2) run a post-process request:
  python test_postprocess.py TARGET.png REF1.png [REF2.png ...] -o out.png

  # health only:
  python test_postprocess.py --health

Exit codes: 0 ok / changed, 3 = 204 no-change, non-zero = error.
This client lives in the PP repo so the main project needs nothing to test.
"""
import argparse
import json
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8005"


def _multipart(fields, files):
    """Build a minimal multipart/form-data body. files: list of (name, filename, bytes)."""
    boundary = "----ppboundary7MA4YWxkTrZu0gW"
    crlf = b"\r\n"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}".encode() + crlf
        body += f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf
        body += value.encode() + crlf
    for name, filename, content in files:
        body += f"--{boundary}".encode() + crlf
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
            + crlf
        )
        body += b"Content-Type: image/png" + crlf + crlf
        body += content + crlf
    body += f"--{boundary}--".encode() + crlf
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def health(url):
    with urllib.request.urlopen(f"{url}/health", timeout=30) as r:
        print(r.read().decode())


def postprocess(url, target, refs, out, meta):
    with open(target, "rb") as f:
        target_bytes = f.read()
    files = [("image", target.split("/")[-1], target_bytes)]
    for i, ref in enumerate(refs):
        with open(ref, "rb") as f:
            files.append(("refs", ref.split("/")[-1], f.read()))
    fields = {}
    if meta:
        fields["meta"] = meta
    body, content_type = _multipart(fields, files)
    req = urllib.request.Request(f"{url}/postprocess", data=body, method="POST")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status = r.status
            data = r.read()
            notes = r.headers.get("X-PP-Notes", "")
            swapped = r.headers.get("X-PP-Swapped-Faces", "?")
            enhanced = r.headers.get("X-PP-Enhanced", "?")
            if status == 204 or not data:
                print("204 No change (nothing applicable).")
                return 3
            with open(out, "wb") as f:
                f.write(data)
            print(f"OK -> {out}  (swapped={swapped}, enhanced={enhanced})")
            if notes:
                print(f"notes: {notes}")
            return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="target image to process")
    ap.add_argument("refs", nargs="*", help="reference face image(s)")
    ap.add_argument("-o", "--out", default="pp_out.png")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--no-enhance", action="store_true", help="disable enhance for this request")
    ap.add_argument("--no-swap", action="store_true", help="disable swap for this request")
    args = ap.parse_args()

    if args.health:
        health(args.url)
        return 0
    if not args.target or not args.refs:
        ap.error("need TARGET and at least one REF (or use --health)")

    options = {}
    if args.no_enhance:
        options["enhance"] = False
    if args.no_swap:
        options["swap"] = False
    meta = json.dumps({"options": options}) if options else ""
    return postprocess(args.url, args.target, args.refs, args.out, meta)


if __name__ == "__main__":
    sys.exit(main())
