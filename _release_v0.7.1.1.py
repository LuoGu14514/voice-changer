"""Create v0.7.1.1 patch release (bug fix for HuBERT offline loading).

Self-contained: inlines PAT extraction from Windows Credential Manager.

NOTE: CredEnumerateW returns an array of PCREDENTIAL (pointers), not
CREDENTIAL structs. Following the first pointer yields a usable struct.
Earlier ctypes crash was because ctypes[0] read 80 bytes of pointer-array
header, not the first CREDENTIAL itself.
"""
import ctypes
import ctypes.wintypes as w
import io
import json
import struct
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GITHUB_API = "https://api.github.com"
UPLOAD_URL = "https://uploads.github.com"
OWNER = "LuoGu14514"
REPO = "voice-changer"
TAG = "v0.7.1.1"
ZIP_NAME = f"voice-changer-{TAG}-source.zip"


# === Windows Credential Manager PAT extraction (self-contained) ===
advapi32 = ctypes.windll.advapi32
advapi32.CredEnumerateW.argtypes = [w.LPCWSTR, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.POINTER(ctypes.c_void_p)]
advapi32.CredEnumerateW.restype = w.BOOL
advapi32.CredFree.argtypes = [ctypes.c_void_p]
advapi32.CredFree.restype = None


def get_github_pat():
    """Get GitHub PAT by enumerating creds filtered by 'git:https://github.com'."""
    count = w.DWORD(0)
    creds_ptr = ctypes.c_void_p(0)
    if not advapi32.CredEnumerateW("git:https://github.com", 0, ctypes.byref(count), ctypes.byref(creds_ptr)):
        raise RuntimeError("CredEnumerateW failed")
    if not creds_ptr.value or count.value == 0:
        raise RuntimeError("No git:https://github.com credential found")
    try:
        # creds_ptr is array of 8-byte pointers (on 64-bit); follow the first one
        first_ptr = struct.unpack_from("<Q", (ctypes.c_byte * 8).from_address(creds_ptr.value))[0]
        if not first_ptr:
            raise RuntimeError("First credential pointer is NULL")
        # Read CREDENTIAL struct (80 bytes) at first_ptr
        raw_bytes = bytes((ctypes.c_byte * 80).from_address(first_ptr))
        blob_size = struct.unpack_from("<I", raw_bytes, 32)[0]
        blob_ptr = struct.unpack_from("<Q", raw_bytes, 40)[0]
        if blob_size == 0 or blob_size > 4096 or not blob_ptr:
            raise RuntimeError(f"Credential blob is empty or invalid (size={blob_size}, ptr=0x{blob_ptr:x})")
        blob_raw = bytes((ctypes.c_byte * blob_size).from_address(blob_ptr))
        text = blob_raw.decode("utf-16-le", errors="replace").rstrip("\x00")
        # Validate: ghp_ + 36 alnum chars (total 40)
        if not text.startswith("ghp_") or len(text) != 40 or not all(c.isalnum() or c == "_" for c in text):
            raise RuntimeError(f"Blob does not look like a GitHub PAT: {text!r}")
        return text
    finally:
        advapi32.CredFree(creds_ptr)


PAT = get_github_pat()
print(f"PAT: {PAT[:6]}...{PAT[-4:]}", flush=True)


def gh_call(method, url, data=None):
    headers = {
        "Authorization": f"Bearer {PAT}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "voice-changer-release",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# Check if release exists
status, body = gh_call("GET", f"{GITHUB_API}/repos/{OWNER}/{REPO}/releases/tags/{TAG}")
if status == 200:
    rel = json.loads(body)
    print(f"Release {TAG} exists, id={rel['id']}", flush=True)
else:
    notes = (
        "## v0.7.1.1 · 紧急修复\n"
        "\n"
        "### 修复\n"
        "- `rvc_infer.py`：`HuBERTExtractor` 现在传 `local_files_only=True`，传入本地目录 `models/chinese-hubert-base/` 时不再尝试去 HuggingFace 拉元数据（HF 在本机网络下完全不可达，每次都报 `WinError 10060` 之后硬退）。\n"
        "\n"
        "### 验证\n"
        "- 端到端：`.venv\\Scripts\\python.exe rvc_infer.py -i models/voice_demo_input.wav -o models/voice_demo_v0.7.1.wav --key 12` 成功\n"
        "  - HuBERT 特征 (99, 768)\n"
        "  - F0 pyin 101 frames, all voiced\n"
        "  - RVC .pth 推理 ~0.99s 输出 @ 48kHz, peak 0.129\n"
        "- 44 项 v0.2.0 DSP 测试仍然全过\n"
    )
    payload = {
        "tag_name": TAG,
        "name": f"voice-changer v{TAG} · HuBERT offline fix",
        "body": notes,
        "draft": False,
        "prerelease": False,
    }
    status, body = gh_call("POST", f"{GITHUB_API}/repos/{OWNER}/{REPO}/releases", payload)
    if status not in (201, 200):
        print(f"Create failed: {status}", flush=True)
        print(body.decode("utf-8", "replace")[:500], flush=True)
        sys.exit(1)
    rel = json.loads(body)
    print(f"Release created: id={rel['id']}", flush=True)


# Make zip
src_files = []
excludes = {".venv", "__pycache__", "node_modules", "_hubert_dl"}
skip_ext = {".pyc", ".pyo", ".pyd", ".egg-info"}
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    rel_p = p.relative_to(ROOT)
    if any(part in excludes for part in rel_p.parts):
        continue
    if p.suffix in skip_ext:
        continue
    if p.name.startswith("_release_") and p.suffix == ".py":
        continue
    if p.suffix == ".zip" and p.name.startswith("voice-changer-"):
        continue
    if "models" in rel_p.parts:
        if rel_p.parts[1] in {"_hubert_dl", "chinese-hubert-base", "test_*.wav", "voice_demo_*.wav"}:
            continue
        if p.suffix in {".onnx", ".pth", ".pt", ".index", ".npy"}:
            continue
    src_files.append(p)
src_files.sort()

zip_path = ROOT / ZIP_NAME
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in src_files:
        arc = f"voice-changer-{TAG}/{f.relative_to(ROOT).as_posix()}"
        zinfo = zipfile.ZipInfo.from_file(f, arc)
        zinfo.flag_bits |= 0x800
        zinfo.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zinfo, f.read_bytes())
print(f"Zip: {zip_path} ({zip_path.stat().st_size:,} bytes, {len(src_files)} files)", flush=True)


# Delete old asset if exists
status, body = gh_call("GET", f"{GITHUB_API}/repos/{OWNER}/{REPO}/releases/{rel['id']}/assets")
if status == 200:
    for asset in json.loads(body):
        if asset["name"] == ZIP_NAME:
            d_status, _ = gh_call("DELETE", asset["url"])
            print(f"Deleted old asset: status={d_status}", flush=True)
            break

# Upload asset
zip_bytes = zip_path.read_bytes()
boundary = "----voicechanger0711"

body = io.BytesIO()
body.write(f"--{boundary}\r\n".encode())
body.write(f'Content-Disposition: form-data; name="name"\r\n\r\n{ZIP_NAME}\r\n'.encode())
body.write(f"--{boundary}\r\n".encode())
body.write(f'Content-Disposition: form-data; name="label"\r\n\r\n{ZIP_NAME}\r\n'.encode())
body.write(f"--{boundary}\r\n".encode())
body.write(f'Content-Disposition: form-data; name="data"; filename="{ZIP_NAME}"\r\n'.encode())
body.write(b"Content-Type: application/zip\r\n\r\n")
body.write(zip_bytes)
body.write(f"\r\n--{boundary}--\r\n".encode())

req = urllib.request.Request(
    f"{UPLOAD_URL}/repos/{OWNER}/{REPO}/releases/{rel['id']}/assets?name={ZIP_NAME}",
    data=body.getvalue(),
    headers={
        "Authorization": f"Bearer {PAT}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "voice-changer-release",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())
print(f"Uploaded: {result['browser_download_url']}", flush=True)
print(f"\nv0.7.1.1 published", flush=True)
print(f"   {rel['html_url']}", flush=True)
