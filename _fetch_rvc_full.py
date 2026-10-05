"""Fetch the entire RVC-Project infer/ folder for local integration."""
import urllib.request
import json
import os
import shutil
from pathlib import Path


BASE = "https://api.github.com/repos/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
RAW = "https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read()
    except Exception as e:
        return None


def list_dir(path):
    data = fetch(BASE + "/contents/" + path)
    if data is None:
        return []
    return json.loads(data)


def fetch_file(rel_path: str, dest: Path):
    data = fetch(RAW + rel_path)
    if data is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def walk(path, out_root: Path, skip=None):
    skip = skip or set()
    items = list_dir(path)
    for it in items:
        if it["type"] == "dir":
            walk(path + "/" + it["name"], out_root, skip)
        else:
            rel = path + "/" + it["name"] if path else it["name"]
            rel = rel.lstrip("/")
            if any(rel.endswith(s) for s in skip):
                continue
            dst = out_root / rel
            ok = fetch_file("/" + rel, dst)
            if ok:
                print(f"  ok  {rel}  ({dst.stat().st_size} B)")
            else:
                print(f"  ERR {rel}")


def main():
    out_root = Path("rvc_lib")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir()

    # We only want infer/ subfolder (and a few extras for completeness)
    print("=== Fetching infer/ ===")
    walk("infer", out_root)

    # Also fetch requirements to know deps
    print("\n=== requirements ===")
    for fname in ("requirments_cpu_py312.txt",):
        fetch_file("/" + fname, Path(fname))
        if Path(fname).exists():
            print(f"  ok {fname} ({Path(fname).stat().st_size} B)")

    print(f"\n=== Files in rvc_lib/ ===")
    total = 0
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            print(f"  {p.relative_to(out_root)}  ({sz} B)")
            total += sz
    print(f"\nTotal: {total} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()