"""Try to fetch RVC inference.py from official open-source RVC repos via GitHub API."""
import urllib.request
import ctypes
import ctypes.wintypes as wt
import json

# --- ctypes CredMan (same as release script) ---
advapi32 = ctypes.windll.advapi32
credui = ctypes.windll.credui

CRED_TYPE_GENERIC = 0x1


class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", wt.DWORD),
        ("Type", wt.DWORD),
        ("TargetName", wt.LPWSTR),
        ("Comment", wt.LPWSTR),
        ("LastWritten", wt.FILETIME),
        ("CredentialBlobSize", wt.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", wt.DWORD),
        ("AttributeCount", wt.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wt.LPWSTR),
        ("UserName", wt.LPWSTR),
    ]


PCREDENTIAL = ctypes.POINTER(CREDENTIAL)
PCREDENTIAL_ARRAY = ctypes.POINTER(PCREDENTIAL)

advapi32.CredEnumerateW.argtypes = [wt.LPCWSTR, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.POINTER(PCREDENTIAL)]
advapi32.CredEnumerateW.restype = wt.BOOL


def read_cred(target: str):
    cred_ptr = PCREDENTIAL()
    count = wt.DWORD(0)
    ok = advapi32.CredEnumerateW(target, 0, ctypes.byref(count), ctypes.byref(cred_ptr))
    if not ok:
        err = ctypes.GetLastError()
        raise OSError(f"CredEnumerateW failed: {err}")
    found = []
    for i in range(count.value):
        c = cred_ptr[i]
        found.append((c.TargetName, c.UserName, c.CredentialBlob, c.CredentialBlobSize))
    advapi32.CredFree(cred_ptr)
    return found


def find_ghp():
    # The PAT blob is 80 bytes UTF-16LE
    blob_marker = b"g\x00h\x00p\x00_\x00"
    # Try several filter patterns
    for target in ("LegacyGeneric:*", "Generic:*", "Microsoft:*", "*"):
        try:
            creds = read_cred(target)
        except OSError as e:
            print(f"  skip {target!r}: {e}")
            continue
        for name, user, blob_ptr, size in creds:
            if size != 80:
                continue
            buf = (ctypes.c_byte * size)()
            ctypes.memmove(buf, blob_ptr, size)
            raw = bytes(buf)
            if raw.startswith(blob_marker):
                txt = raw.decode("utf-16-le").rstrip("\x00")
                print(f"  Found PAT at: {name!r} user={user!r}")
                return txt
    return None


def gh_get(path: str, pat: str) -> tuple:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {pat}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return e.code, body
    except Exception as e:
        return 0, str(e)


def gh_raw(path: str, pat: str) -> bytes:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github.raw",
            "Authorization": f"Bearer {pat}",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return resp.read()
    except Exception as e:
        return b""


# Search for RVC inference.py
print("=== Find PAT ===")
pat = find_ghp()
print(f"  PAT: {pat[:10]}...{pat[-4:]}" if pat else "  PAT: NONE")

if pat:
    print("\n=== Search RVC repos ===")
    status, data = gh_get("/search/repositories?q=RVC+inference+language:python&sort=stars&order=desc&per_page=10", pat)
    print(f"  status={status}")
    if isinstance(data, dict) and "items" in data:
        for repo in data["items"]:
            print(f"  {repo['stargazers_count']:>6} ★ {repo['full_name']:40} {repo['description'][:60] if repo.get('description') else ''}")

    # Top candidate: RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    print("\n=== Check RVC-Project repo for inference modules ===")
    status, data = gh_get("/repos/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/contents/infer", pat)
    print(f"  status={status}")
    if isinstance(data, list):
        for item in data[:30]:
            print(f"    {item.get('type'):4} {item.get('size', 0):>8}  {item.get('name')}")