"""Create the v0.8.0 GitHub release.

Self-contained. The asset zip is produced with `git archive`, so .gitignore
is respected automatically and the 12 GB of local model/venv directories
never leak into the archive.

PAT is read from the local git remote URL (no hardcoded secret, no
Credential Manager ctypes dance).
"""
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GIT = r"C:\ProgramData\Microsoft\Windows\Templates\G10_Multi-Mode\Git\cmd\git.exe"
GITHUB_API = "https://api.github.com"
UPLOAD_URL = "https://uploads.github.com"
OWNER = "LuoGu14514"
REPO = "voice-changer"
TAG = "v0.8.0"
ZIP_NAME = f"voice-changer-{TAG}-source.zip"

NOTES = """## v0.8.0 — IndexTTS-2.5 本地 TTS

从 GPT-SoVITS 切到 **IndexTTS-2.5** 做角色音色克隆，纯 CPU 推理。

### 新增

| 文件 | 作用 |
|---|---|
| `ind_extts_cli.py` | 命令行合成入口：5 语言、语速、扩散步数、峰值保护 |
| `indextts_worker.py` | 常驻合成进程，JSON-Lines 协议，模型只加载一次 |
| `tts_app.py` | tkinter GUI：角色 + 情绪参考下拉、文本、参数、试听、另存为 |
| `INDEXTTS_README.md` | 环境 / 模型 / 用法 / 性能 / 源码改动 / 踩坑记录 |

### 架构

```
tts_app.py (Python 3.12, tkinter)  --JSON-Lines-->  indextts_worker.py (Python 3.11)
                                                     IndexTTS-2.5 常驻内存
```

模型加载 28–42 秒，常驻复用，GUI 启动时就在后台拉起来。

### 性能（Ryzen 7 7840HS，无 CUDA）

- 单句 3–4 秒音频合成 **34–60 秒**（RTF 11–19）
- 扩散步数扫描 8 / 10 / 15 / 20 / 25，**默认取 20**：比上游 25 快 24%，峰值 0.759 无削波
- `--num-beams 1`（上游 3）
- 16 线程（torch 默认只用物理核数）

### 两个硬约束

1. **必须用 Python 3.11** —— 上游 `requires-python = ">=3.10,<3.12"`，独立 venv
2. **必须在 ASCII 路径下运行** —— `kaldifst` 的 C++ `std::ifstream` 打不开中文路径，
   项目里建了 `D:\\vc` junction 绕过

### 峰值保护

不同参考音频响度差异会让输出电平漂移（实测青雀某段输出峰值顶到 1.000 有削波）。
CLI 和 worker 都会在写盘前检查峰值，超过 0.95 就整体缩放，只动电平不动音色。

### 实测

| 角色 | 文本 | 时长 | peak | 削波 | RTF |
|---|---|---|---|---|---|
| 三月七 | 原来你也在这里呀。今天的星穹列车，也要一起出发吗？ | 3.87 s | 0.870 | 0.00% | 11.53 |
| 青雀 | 太卜大人说，今日卦象大吉。要不要来一局？ | 4.23 s | 0.950 | 0.00% | 17.57 |

### 清理

删除 GPT-SoVITS 路线的 10 个探测脚本与 `gsv_infer.py`（依赖始终装不上，已被取代）。

---

**完整源码**见附件 zip（`git archive` 生成，不含模型与 venv）。
IndexTTS 模型与本项目依赖需按 `INDEXTTS_README.md` 自行准备。
"""


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
        "name": f"{TAG} — IndexTTS-2.5 本地 TTS",
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
