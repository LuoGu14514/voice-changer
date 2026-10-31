"""Create the v0.10.0 GitHub release.

Self-contained. The asset zip is produced with `git archive`, so .gitignore
is respected automatically and the local model/venv directories never leak
into the archive.

PAT is read from the local git remote URL (no hardcoded secret, no
Credential Manager ctypes dance).

Network note: on this machine the geo-DNS answer for **github.com**
(20.205.243.166) is unroutable, so `git push` needs
`git -c http.curloptResolve=github.com:443:<working edge ip>`.
The **API hosts are fine with normal DNS** (api.github.com ->
20.205.243.168, uploads.github.com -> 20.205.243.161), so this script must
NOT remap them: the ".5"/".6" edges answer the asset-upload route with
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
TAG = "v0.10.0"
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

NOTES = """## v0.10.0 — 自定义角色：零样本克隆 + 云端音色库

用户问了一句：**「不可以自定义角色吗」**。

之前那 30 个角色是「原项目角色表 → MiniMax 官方系统音色」的手工映射，
每一项都写死在 `MINIMAX_PRESETS` 里。这一版把「自己造角色」真正做出来。

### 🔍 先探权限，再写代码

拿手上的订阅 Key 逐个打接口，得到一组决定性的事实：

| 接口 | 结果 | 含义 |
|---|---|---|
| `/v1/get_voice`（voice_cloning） | 账号里已有 **1 个**克隆音色 | 之前建过 |
| `/v1/get_voice`（voice_generation） | 0 个 | 没建过 |
| `/v1/voice_clone` | **2061** `your current token plan not support model, voice_clone` | 套餐不支持新建克隆 |
| `/v1/voice_design` | **2061** `... not support model, voice_design` | 套餐不支持新建设计 |
| `/v1/t2a_v2`（用已有克隆音色） | ✅ 50676 bytes / 3060 ms | **已有的自定义音色照常能用** |

所以「自定义角色」不是一个功能，是**三条路**，各自边界不同。

### 🧬 三条路

| 方式 | 需要什么 | 得到什么 | 本机可行 |
|---|---|---|---|
| **零样本参考音频** | 一段 3-15 秒干净语音 | 一个新角色，音色来自这段音频 | **可行**（已实测跑通） |
| **拉取云端自定义音色** | 账号里已有的克隆/设计音色 | 把它们加进角色库 | **可行**（已实测跑通） |
| **声音克隆 / 声音设计** | 支持该能力的套餐 | 云端新建音色 | 被 2061 挡住（入口保留 + 明确提示） |

### 🎯 IndexTTS 零样本：唯一真正「零训练」的路

IndexTTS 的 `spk_audio_prompt` 本来就是给零样本克隆设计的——引擎侧一直支持
传任意 `ref_audio`，缺的只是 GUI 上一个「选文件」的入口。

GUI 新增「自定义角色」面板：

- 「浏览…」选参考音频 → 填角色名 → 「保存为角色」
- 角色出现在 **IndexTTS 后端**的角色下拉里，选中直接合成
- 「拉取云端音色」把账号里的 `voice_cloning` / `voice_generation` 全列出来，一键入库
- 「声音克隆…」/「声音设计…」保留入口，遇到 2061 会明确提示套餐限制并给出替代方案

### 💾 角色库存哪儿

`%APPDATA%\\voice-changer\\custom_voices.json`，纯 JSON，随时手改：

```json
{
  "version": 1,
  "voices": [
    {
      "name": "我的角色",
      "group": "自定义",
      "backend": "indextts",
      "voice_id": "",
      "ref_audio": "D:\\\\音频\\\\我的角色.wav",
      "speed": 1.0, "pitch": 0, "emotion": "",
      "source": "local",
      "created": "2026-10-31 20:00:00"
    }
  ]
}
```

`backend` 决定它出现在哪个后端的角色列表里：
`indextts`（有 `ref_audio`）走本地零样本，`minimax`（有 `voice_id`）走云端。

### 🔧 顺手修的三个 bug

**1. `upload_file()` 取不到 file_id**

实测 MiniMax 的响应是**嵌套**的：

```json
{"file": {"file_id": 458331960050266, ...}, "base_resp": {...}}
```

原实现取顶层 `obj.get("file_id")` → `None` → 抛「上传成功但没拿到 file_id」。
原来的 mock 测试也按顶层写，所以**测试全绿但真实调用必挂**。
已修为嵌套优先、顶层兜底，并补了一条用真实响应结构的新用例。

**2. GUI 走的 worker 没吃到「吞尾音」修复**

`indextts_worker.py`（GUI 实际走的那条路）和 `indextts_server.py` 各自写了一份
`IndexTTS2` 加载代码，而 `indextts_engine.py` 里已经有封装好的 `IndexTTSEngine`。
三份实现 → 三种行为：

| 能力 | engine | server | worker |
|---|---|---|---|
| `torch.load(mmap=True)` | ✅ | ✅ | ❌ |
| ASCII 路径别名 | ✅ | ✅ | ❌ |
| 语义模型 bf16 | ✅ | ✅ | ❌ |
| **末尾淡出（修吞尾音）** | ✅ | ✅ | **❌** |

最后一条是真 bug：v0.8 那次「吞尾音」的修复只加在 `indextts_engine.py` 上，
**GUI 实际用的 worker 根本没走那段代码**——所以 GUI 里 IndexTTS 合成出来的东西
尾音还是会被切。这次把 worker 改成直接用 `IndexTTSEngine`，一次性对齐。

**3. 角色名带「·」取错**

下拉框显示格式是「分组 · 名字」，原来用 `split("·")[-1]` 取名字，
角色名里自己带「·」就取错了。改成按已知分组前缀剥离。
另外补了 `safe_filename()` 清掉 Windows 文件名保留字符
（`< > : " / \\ | ? *`），自定义角色名下拼输出路径不再炸。

### 💥 顺手定位到 IndexTTS 加载崩溃的真凶

之前一直定性为「提交量耗尽」，写了 bf16 补丁但**从没验证过**（更早阶段就崩了）。
这次重试：

| 条件 | 结果 |
|---|---|
| `semantic_bf16=False`（fp32），默认线程 | `exit=-1073741819`（AV，一如既往） |
| `semantic_bf16=True`，`threads=4` | **exit=0**，加载成功，合成成功 |

commit 额度当时是 7423 MB 余量，引擎 fp32 稳态要 ~7.5 GB——就差那么一点，
fp32 正好卡在悬崖边上；bf16 把语义模型 w2v-bert 从 2.2 GB 压到 ~1.1 GB，就过去了。

于是把 `semantic_bf16` 默认值**翻成 `True`**（engine + server 同步），
命令行开关从 `--semantic-bf16` 改成 `--no-semantic-bf16`（默认开）。

### 📊 实测数据

**MiniMax 自定义角色**（账号里已有的克隆音色）：

```
云端自定义音色: [('NahidaDesktopTest_20260812_29bc0d2a', '克隆')]
已加入角色库: 娜希达测试 → NahidaDesktopTest_20260812_29bc0d2a
合成 OK: 213674 bytes  1.06s  时长 3.32s
分组: [('原',1), ('女',8), ('男',7), ('童',7), ('特效',7), ('自定义',1)]
```

**IndexTTS 零样本自定义角色**（10 秒参考音频，`semantic_bf16=True`, `threads=4`）：

```
load 成功 → 合成成功
Total inference time: 141.94 seconds
Generated audio length: 5.78 seconds
输出: 6.03s / 22050 Hz / 单声道 / 16-bit / 266044 bytes
peak 0.950  rms 0.1722
尾 100ms peak = 0.0000   ← 淡出生效，没有咔哒声
```

**worker 路径（GUI 实际走的那条）复测**：

```
ready: load_s 69.7  sr 22050
resp: ok=True  dur 9.712  peak 0.8662  infer_s 128.44  rtf 13.22
check: tail100_peak = 0.0000
```

RTF ~13（CPU 纯推理）——本地后端是「准离线」，日常还是 MiniMax 云端顺手。

### 🧪 测试：70 项 → 88 项，全部离线

```bat
python -m pytest test_minimax_tts.py test_tts_app.py test_voice_core.py -v
```

新增 18 项：自定义角色库读写 / 校验 / 同名覆盖 / 坏 JSON 容错 / 分组追加 /
merged / 云端音色列举 / 嵌套 file_id / GUI 面板控件 / 自定义角色进下拉 /
`_current_ref` 取自定义音频 / 后端错配拦截 / Treeview 列表 /
`name_from_label` / `safe_filename`。

全部用本地 `ThreadingHTTPServer` mock，**不发外网请求、不需要 API Key**。

顺带修了一处测试污染：测试现在把自定义角色库指到临时文件，
**不再读用户真实的 `%APPDATA%`**——否则用户在 GUI 里加过角色，
「角色下拉有 30 项」的断言就会炸。

### 📚 文档

- `README.md`：新增「版本历程」表、「自定义角色（v0.10.0 新增）」小节
- `EXPERIENCE.md` §15：完整记录权限探测、三条路的取舍、bf16 定位过程、
  worker/engine 重复实现的教训、角色库设计、嵌套响应的坑

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
        "name": f"{TAG} — 自定义角色：零样本克隆 + 云端音色库",
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
