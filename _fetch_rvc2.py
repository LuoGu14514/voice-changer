"""Download the official RVC inference code from RVC-Project GitHub.

Goal: get hubert.py + module/ + vc/ + necessary infra so we can run RVC v2 inference locally.
"""
import urllib.request
import json
import os
import sys
from pathlib import Path


BASE = "https://api.github.com/repos/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
RAW = "https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read()
    except urllib.error.HTTPError as e:
        print(f"  ERR {e.code} for {url}")
        return None
    except Exception as e:
        print(f"  ERR {e} for {url}")
        return None


def list_dir(api_path):
    """List repo directory contents."""
    data = fetch(BASE + api_path)
    if data is None:
        return []
    return json.loads(data)


def fetch_raw(rel_path: str, dest: Path):
    """Download a raw file from main branch."""
    data = fetch(RAW + rel_path)
    if data is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"  fetched {rel_path} ({len(data)} bytes)")
    return True


def main():
    out_root = Path("_rvc_src")
    if out_root.exists():
        # Clean previous fetch
        import shutil
        shutil.rmtree(out_root)
    out_root.mkdir()

    # 1. infer/modules/ contents
    print("=== infer/modules/ ===")
    items = list_dir("/contents/infer/modules")
    module_files = []
    for it in items:
        if it["type"] == "file":
            print(f"  file {it['size']:>8} {it['name']}")
            module_files.append(it["name"])
        elif it["type"] == "dir":
            print(f"  dir  {it['name']}/")
            # list subdir
            sub_items = list_dir(it["_links"]["href"].replace("https://api.github.com", ""))
            for s in sub_items:
                print(f"    file {s.get('size', 0):>8} {s.get('name')}")
                if s["type"] == "file":
                    fetch_raw(s["path"], out_root / Path(s["path"]).relative_to("infer/modules"))

    # 2. infer/modules/vc/ (inference wrapper)
    print("\n=== infer/modules/vc/ ===")
    items = list_dir("/contents/infer/modules/vc")
    for it in items:
        if it["type"] == "file":
            print(f"  file {it['size']:>8} {it['name']}")
            fetch_raw(it["path"], out_root / Path(it["path"]).relative_to("infer/modules"))
        elif it["type"] == "dir":
            print(f"  dir  {it['name']}/")

    # 3. main infer/cli.py + key helpers (for Shared class etc.)
    print("\n=== infer/ top level ===")
    items = list_dir("/contents/infer")
    for it in items:
        if it["type"] == "file":
            print(f"  file {it['size']:>8} {it['name']}")

    # Show summary
    print(f"\n=== Files downloaded to {out_root}/ ===")
    total = 0
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_root)}  ({p.stat().st_size} bytes)")
            total += p.stat().st_size
    print(f"\nTotal: {total} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()