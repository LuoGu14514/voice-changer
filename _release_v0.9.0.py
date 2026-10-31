"""Create the v0.9.0 GitHub release.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GIT = r"C:\ProgramData\Microsoft\Windows\Templates\G10_Multi-Mode\Git\cmd\git.exe"
GITHUB_API = "https://api.github.com"
UPLOAD_URL = "https://uploads.github.com"
OWNER = "LuoGu14514"
REPO = "voice-changer"
TAG = "v0.9.0"
ZIP_NAME = f"voice-changer-{TAG}-source.zip"

NOTES = """## v0.9.0 — MiniMax 云端 TTS 后端 + 30 角色预设映射

在 v0.8.0 的 IndexTTS 本地克隆之上，加了一个**云端后端**。
`minimax_tts.py` 是纯标准库写的 MiniMax T2A v2 客户端，**零第三方依赖**。

### 新增

| 文件 | 作用 |
|---|---|
| `minimax_tts.py` | MiniMax T2A v2 客户端：同步 HTTP / SSE 流式 / WS 流式 / 双向流式 / 异步长文本 / 声音克隆 / 声音设计 / 音色管理 |
| `docs/MINIMAX_TTS_API.md` | 完整 API 参考（1003 行 / 17 章）：参数表、错误码表、限流、定价、332 条系统音色表 |
| `test_minimax_tts.py` | 40 项离线测试（本地 mock HTTP server，不需要 API Key） |
| `test_tts_app.py` | 9 项 GUI 冒烟测试 |

### 双后端

`tts_app.py` 顶部一个单选框切换：

| 后端 | 延迟 | 内存 | 依赖 | 适合 |
|---|---|---|---|---|
| **MiniMax 云端**（默认） | 几秒 | ~0 | 只需 API Key | 日常使用 ✅ |
| **IndexTTS-2.5 本地** | 30–60 s | ~7.5 GB | `_indextts_venv` + 5.4 GB 模型 | 离线 / 隐私敏感 |

> **IndexTTS worker 不再开机自动拉起**。以前 GUI 一启动就把 7.5 GB 的
> 推理进程拽起来，加载 30–70 秒，用户想用云端也躲不掉。现在改成
> 选中该后端后首次点「合成」才加载。

### 30 个角色预设

原项目的 30 个角色（真女声 / 御姐 / 大叔 / 萝莉 / 汤姆猫 / 机器人 / 电话音 …）
在 MiniMax 后端里一一对应到官方系统音色 + 语速 + 音高 + 情感 + `voice_modify` 特效：

| 组 | 角色 |
|---|---|
| 原 | 原声 |
| 女 | 真女声、御姐、客服女、播音女、少妇、老奶奶、萌妹、嗲嗲 |
| 男 | 真男声、大叔、磁性男、恶魔、老人、客服男、播音男 |
| 童 | 萝莉、正太、小孩、娃娃音、小猫、花栗鼠、汤姆猫 |
| 特效 | 机器人、电音女王、外星人、兽人、幽灵、回声、电话音 |

`voice_modify.sound_effects` 有 4 个官方取值（`spacious_echo` / `auditorium_echo` /
`lofi_telephone` / `robotic`），正好覆盖「幽灵 / 回声 / 电话音 / 机器人」这四个。

### Tk 线程安全

合成放在后台线程，但**所有控件的值必须在 GUI 线程读出来再传进去**。
第一版在 worker 里直接 `widget.get()`，这是定时炸弹，现在改掉了。

### 顺手修的两个老问题

**1. 吞尾音。** IndexTTS 的 `target_lengths = code_lens * 1.72`，`code_lens` 由
`stop_mel_token` 随机采样决定，停止符落得偏早就会把元音尾巴切掉，
末尾还留一个孤立的 step（click）。上游 `utils/common.py` 有 `fade_out_tail`
但 `infer_v2_5.py` 走 `save_pcm_wav()` 没调它。现在在
`indextts_engine.synthesize` 里自己做 **20 ms 升余弦淡出 + 250 ms 静音 padding**：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 末 100 ms Peak / RMS | 1797 / 239 | **0 / 0** |
| 尾静音 | 80.7 ms | **361.2 ms** |
| 末活跃点 | t−50 ms（被切） | **t−4.08 s** |

**2. 内存提交量耗尽。** 这台机器物理内存 15.2 GB、系统提交量长期 28 GB 上下，
IndexTTS 加载期会撞上限（`ACCESS_VIOLATION`）。`indextts_engine.py` 加了两处缓解：

- `_enable_mmap_torch_load()`：`torch.load` 对 ≥64 MB 的 `.pth/.ckpt` 走 `mmap=True`。
  `gpt.pth` 3.1 GB，默认加载会反序列化出一整份副本再拷进模型，
  峰值多 3.1 GB 提交量。实测从崩溃变成 0.6 s 成功。
- `_ascii_alias()`：把含中文的路径换成指向同目标的 ASCII junction（`D:\\vc`）。
  torch 的 mmap 走 `CreateFileMapping`，在中文路径上有「映射建好但指针不可用」
  的踩坑表现，`load_state_dict` 里 `copy_` 直接 AV。

> 这两招提高成功率，但**根治还是要腾内存**。所以默认后端改成云端。

### 客户端 API

```python
from minimax_tts import MiniMaxTTS

tts = MiniMaxTTS(api_key="sk-xxx")
tts.synth_to_file("你好", "Chinese (Mandarin)_Sweet_Lady", "hi.wav")
for chunk in tts.synth_stream("你好", "Chinese (Mandarin)_Sweet_Lady"):
    ...
tts.clone_voice("ref.wav", "my-voice-id")
```

Key 存在 `%APPDATA%\\voice-changer\\minimax.json`，也支持环境变量 `MINIMAX_API_KEY`。

### 测试

```bat
python -m pytest test_minimax_tts.py test_tts_app.py -v
```

49 项，全部离线（本地 mock HTTP server），不发外网请求、不需要 API Key：
请求体构造、20 个官方错误码解析、multipart 上传格式、异步接口字段名
（`audio_sample_rate`，跟同步接口不一样）、克隆 voice_id 校验、
30 个预设与原项目一致（名字 + 分组）、GUI 后端切换等。

---

**完整源码**见附件 zip（`git archive` 生成，不含模型与 venv）。
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
        "name": f"{TAG} — MiniMax 云端 TTS 后端 + 30 角色预设",
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
