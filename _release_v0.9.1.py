"""Create the v0.9.1 GitHub release.

Self-contained. The asset zip is produced with `git archive`, so .gitignore
is respected automatically and the 12 GB of local model/venv directories
never leak into the archive.

PAT is read from the local git remote URL (no hardcoded secret, no
Credential Manager ctypes dance).

Network note: on this machine the geo-DNS answer for **github.com**
(20.205.243.166) is unroutable, so `git push` needs
`git -c http.curloptResolve=github.com:443:<working edge ip>`.
The **API hosts are fine with normal DNS** (api.github.com ->
20.205.243.168, uploads.github.com -> 20.205.243.161), so this script must
NOT remap them: the `.5`/`.6` edges answer the asset-upload route with
404 "Not Found", and forcing them breaks the upload.
"""
import json
import re
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GIT = r"C:\ProgramData\Microsoft\Windows\Templates\G10_Multi-Mode\Git\cmd\git.exe"
GITHUB_API = "https://api.github.com"
UPLOAD_URL = "https://uploads.github.com"
OWNER = "LuoGu14514"
REPO = "voice-changer"
TAG = "v0.9.1"
ZIP_NAME = f"voice-changer-{TAG}-source.zip"

# Fallback edge addresses for GitHub API hosts (geo-DNS answers may be dead).
# NOTE: the ".3" addresses are web-only edges: they answer api.github.com with
# 301 -> https://github.com/... . The API edge answers on ".5"/".6".
FALLBACK_IPS = [
    "140.82.112.5",
    "140.82.113.5",
    "140.82.114.5",
    "140.82.116.5",
    "20.205.243.168",
    "20.27.177.116",
]
API_HOSTS = ("api.github.com", "uploads.github.com")

NOTES = """## v0.9.1 — MiniMax Key 域名自动回退 + 真 Key 全量实测

v0.9.0 把 MiniMax 云端 TTS 接进来了，但**从没拿真 Key 跑过**。
这一版是拿到订阅 Key 之后的第一轮实测与修复。

### 🔑 最重要的发现：Key 和域名绑定

同一把 Key，不同域名结果完全相反：

| 域名 | `/v1/get_voice` | `/v1/t2a_v2` | `/v1/text/chatcompletion_v2` |
|---|---|---|---|
| `api.minimax.io`（新平台） | **2049 invalid api key** | 2049 | 2049 |
| `api.minimaxi.com`（老平台） | ✅ | ✅ | ✅ |
| `api.minimax.chat` | ✅ | ✅ | ✅ |

`.io` 和 `.com` 是**两套互不相通的 Key 命名空间**。v0.9.0 里写死了
`https://api.minimax.io`，等于对一半用户直接不可用。

### 修复：域名自动探测 + 回退

`minimax_tts.py` 默认 `base_url="auto"`，候选域名按顺序试：

```
https://api.minimaxi.com → https://api.minimax.io → https://api.minimax.chat
```

- 收到 `code == 2049`（invalid api key）→ 自动换下一个域名重试
- 所有域名都失败 → 抛出最后一个的 `MiniMaxError`，带 `code` / `trace_id`
- 用户在 GUI 里显式选了域名 → **不再回退**（选了就是选了）
- HTTP / SSE 流式 / WS 流式 / 双向流式 / 异步 / 克隆 / 设计 / 音色管理
  **全部走同一套回退逻辑**

`minimax_tts.py` 的 `BASE_URLS`、`requested_base_url`、`_next_host()`、
`_try_hosts()` 是新增的公开面。

### 实测：30 个角色预设 30/30 全部真实合成成功

用订阅 Key 逐个跑：

| 指标 | 结果 |
|---|---|
| 成功率 | **30 OK / 0 FAIL** |
| 单条耗时 | 0.78 – 2.02 s（平均 **1.09 s**） |
| 输出格式 | WAV / 32000 Hz / 单声道 / 16-bit |
| 官方系统音色总数 | **303** 条（新旧两套命名空间都有） |

`voice_modify.sound_effects` 那 4 个音效预设也验证过：

| 预设 | 音效 |
|---|---|
| 机器人 | `robotic` |
| 幽灵 | `spacious_echo` |
| 回声 | `auditorium_echo` |
| 电话音 | `lofi_telephone` |

### 计费看 `usage_characters`，不是 `word_count`

10 个中文字符的请求，`usage_characters` 报 **18**，`word_count` 报 **10**。
按 Unicode 码点计。`synth(..., with_meta=True)` 可以直接拿到完整响应体，
`extra_info` 里还有 `audio_length` / `audio_size` / `bitrate` /
`invisible_character_ratio`。

### GUI 新增「域名」下拉框

MiniMax 面板多一行：`自动探测` + 三个候选域名。「测试连接」会把**实际生效的域名**
打进日志，不用猜。Key 可以从 `%APPDATA%\\voice-changer\\minimax.json` 读，
也可以直接粘进输入框。

### 测试：56 项 → 70 项，全部离线

```bat
python -m pytest test_minimax_tts.py test_tts_app.py test_voice_core.py -v
```

新增 7 项专门测回退逻辑：

| 用例 | 验证 |
|---|---|
| `test_default_base_url_is_auto` | 默认走自动探测 |
| `test_base_urls_cover_minimaxi_and_io` | 候选域名表包含 `.com` 和 `.io` |
| `test_auto_host_falls_back_to_next_host_on_2049` | 首个域名 2049 → 换下一个成功 |
| `test_explicit_base_url_does_not_fall_back` | 显式域名不偷跑回退 |
| `test_all_hosts_bad_raises_2049` | 全挂时抛 2049 |
| `test_synth_with_meta_returns_response` | `with_meta=True` 返回 `(bytes, dict)` |
| `test_synth_without_meta_returns_bytes` | 默认仍只返回 bytes |

全部用本地 `ThreadingHTTPServer` mock，**不发外网请求、不需要 API Key**。

### 顺手修的测试污染

`test_tts_app.py::test_missing_api_key_raises` 一开始是挂的：开发机
`%APPDATA%\\voice-changer\\minimax.json` 里存着真 Key，`_load_mm_key()`
会自动填进输入框，测试只 monkeypatch 了 `CONFIG_PATH` 没清 entry。
现在测试里显式 `ent_key.delete(0, "end")`。

### 目录结构的坑（记在 EXPERIENCE.md 里）

`D:\\代码` 本身是个 git 仓库（**还没有任何 commit**），里面混着几十个别的项目；
`D:\\代码\\voice_changer` 是**另一个独立仓库**，才是本项目。
`git status` 一定要带 `-C "D:\\代码\\voice_changer"`，别在父仓库上操作。

### 客户端 API

```python
from minimax_tts import MiniMaxTTS

tts = MiniMaxTTS(api_key="sk-xxx")            # 自动探测域名
tts = MiniMaxTTS(api_key="sk-xxx",
                 base_url="https://api.minimaxi.com")   # 指定域名，不回退

tts.synth_to_file("你好", "Chinese (Mandarin)_Sweet_Lady", "hi.wav")
for chunk in tts.synth_stream("你好", "Chinese (Mandarin)_Sweet_Lady"):
    ...
tts.clone_voice("ref.wav", "my-voice-id")
```

---

**完整源码**见附件 zip（`git archive` 生成，不含模型与 venv）。
"""


def _api_edge_ok(ip):
    """True when this edge really serves the GitHub API (200, not 301)."""
    import http.client
    try:
        s = socket.create_connection((ip, 443), timeout=6)
        ctx = ssl.create_default_context()
        s = ctx.wrap_socket(s, server_hostname="api.github.com")
        c = http.client.HTTPSConnection("api.github.com", timeout=15, context=ctx)
        c.sock = s
        c.request("GET", "/rate_limit",
                  headers={"User-Agent": "probe", "Host": "api.github.com"})
        r = c.getresponse()
        r.read(64)
        return r.status == 200
    except Exception:
        return False


def force_github_ip():
    """Best-effort remap of the GitHub API hosts to a reachable edge IP.

    Normally a no-op: on this network the API hosts answer with the normal
    geo-DNS addresses. The remap only kicks in if a plain request to
    api.github.com fails outright, and it is deliberately limited to
    api.github.com -- remapping uploads.github.com breaks asset uploads
    (those edges answer the upload route with 404 "Not Found").
    """
    original = socket.getaddrinfo

    try:
        s = socket.create_connection(("api.github.com", 443), timeout=8)
        s.close()
        print("[net] api.github.com reachable via normal DNS")
        return
    except Exception:
        pass

    chosen = None
    for ip in FALLBACK_IPS:
        try:
            s = socket.create_connection((ip, 443), timeout=6)
            s.close()
        except Exception:
            continue
        if _api_edge_ok(ip):
            chosen = ip
            break
    if not chosen:
        print("[net] no verified GitHub API edge IP found, using DNS")
        return
    print(f"[net] forcing api.github.com -> {chosen}")

    def patched(host, port, *args, **kwargs):
        if host == "api.github.com":
            return original(chosen, port, *args, **kwargs)
        return original(host, port, *args, **kwargs)

    socket.getaddrinfo = patched


def git(*args, check=True):
    r = subprocess.run([GIT, *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit(f"[git] {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def get_pat():
    url = subprocess.run([GIT, "remote", "get-url", "origin"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    m = re.search(r"ghp_[A-Za-z0-9]{36}", url)
    if not m:
        m = re.search(r"github_pat_[A-Za-z0-9_]{60,}", url)
    if not m:
        sys.exit("[err] no PAT in git remote URL")
    return m.group(0)


def api(method, path, pat, payload=None, raw=False, ctype="application/json"):
    url = path if path.startswith("http") else GITHUB_API + path
    data = None
    if payload is not None:
        data = payload if raw else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {pat}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "voice-changer-release")
    if data is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = r.read()
            return r.status, (json.loads(body) if body and not raw else body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, body


def build_zip():
    out = ROOT / ZIP_NAME
    if out.exists():
        out.unlink()
    # git archive respects tracked files only -> no models, no venvs
    r = subprocess.run(
        [GIT, "archive", "--format=zip", f"--prefix=voice-changer-{TAG}/",
         "-o", str(out), "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"[zip] git archive failed: {r.stderr}")
    print(f"[zip] {out.name}  {out.stat().st_size} bytes")
    return out


def main():
    force_github_ip()

    pat = get_pat()
    print(f"[auth] PAT {pat[:8]}...{pat[-4:]}")

    print("[git] check tag pushed")
    tags = git("tag", "-l", TAG)
    if not tags:
        sys.exit(f"[err] tag {TAG} missing locally")

    zp = build_zip()

    print(f"[api] create release {TAG}")
    status, body = api("POST", f"/repos/{OWNER}/{REPO}/releases", pat, {
        "tag_name": TAG,
        "name": f"{TAG} — MiniMax Key 域名自动回退 + 真 Key 全量实测",
        "body": NOTES,
        "draft": False,
        "prerelease": False,
    })
    print(f"[api] status {status}")
    if status == 201:
        rel = body
        print(f"[api] release id={rel['id']}")
    elif status == 422 and "already_exists" in str(body):
        print("[api] release exists, fetching")
        s2, rel = api("GET", f"/repos/{OWNER}/{REPO}/releases/tags/{TAG}", pat)
        if s2 != 200:
            sys.exit(f"[err] fetch release failed {s2}: {rel}")
        print(f"[api] release id={rel['id']}")
    else:
        sys.exit(f"[err] create release failed: {body}")

    upload_url = (f"{UPLOAD_URL}/repos/{OWNER}/{REPO}/releases/"
                  f"{rel['id']}/assets?name={ZIP_NAME}")
    print(f"[api] upload {ZIP_NAME}")
    status, body = api("POST", upload_url, pat, zp.read_bytes(),
                       raw=True, ctype="application/zip")
    print(f"[api] upload status {status}")
    if status not in (201, 422):
        sys.exit(f"[err] upload failed: {body}")
    if status == 201:
        try:
            meta = json.loads(body)
            print(f"[api] asset id={meta.get('id')}  size={meta.get('size')}")
        except Exception:
            print(f"[api] asset uploaded ({len(body)} bytes of response)")
    else:
        print("[api] asset already exists, skipping")

    print(f"\nOK -> https://github.com/{OWNER}/{REPO}/releases/tag/{TAG}")
    zp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
