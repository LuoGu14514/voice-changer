# -*- coding: utf-8 -*-
"""IndexTTS-2.5 常驻合成 worker。

模型加载要 28-40 秒，所以 GUI 不能每次合成都重启进程。
这个 worker 只加载一次，之后在 stdin/stdout 上用 JSON-Lines 收发请求。

用法（必须用 _indextts_venv 的 python，且工作目录为 D:\\vc）：

    D:\\vc\\_indextts_venv\\Scripts\\python.exe -u indextts_worker.py

协议：
    启动完成  ->  {"ready": true, "load_s": 28.5, "sr": 22050}
    请求      <-  {"id": 1, "text": "...", "ref": "...", "lang": "ZH",
                   "output": "...", "duration_factor": 1.0,
                   "diffusion_steps": 20, "num_beams": 1}
    成功响应  ->  {"id": 1, "ok": true, "output": "...", "dur": 4.23,
                   "rtf": 11.5, "peak": 0.95, "infer_s": 44.6}
    失败响应  ->  {"id": 1, "ok": false, "error": "..."}
    退出      <-  {"cmd": "quit"}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

# 模型内部有大量 print，会污染 stdout 上的 JSON 协议。
# 把 stdout 换到 stderr，协议走独立的 PROTO 句柄。
PROTO = sys.stdout
sys.stdout = sys.stderr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_indextts_src")
CHECKPOINTS = os.path.join(SRC, "checkpoints")

# 让 indextts 包可导入（和 ind_extts_cli.py 保持一致的环境变量设置）
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("HF_HUB_CACHE", os.path.join(CHECKPOINTS, "hf_cache"))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(HERE, "_ms_cache"))
os.environ.setdefault("USE_MODELSCOPE", "true")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, "_cache", "matplotlib"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "_cache", "hf"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(HERE, "_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _d in (
    os.environ["MODELSCOPE_CACHE"],
    os.environ["MPLCONFIGDIR"],
    os.environ["HF_HOME"],
    os.environ["XDG_CACHE_HOME"],
):
    os.makedirs(_d, exist_ok=True)

DEFAULT_DIFFUSION_STEPS = 20
DEFAULT_CFG_RATE = 0.7


def _send(obj):
    PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO.flush()


def _err(msg, exc=None):
    if exc is not None:
        msg = f"{msg}: {type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    _send({"ok": False, "error": msg})


def main():
    # 加载逻辑全部交给 indextts_engine.IndexTTSEngine：
    #   · torch.load 走 mmap=True，省掉 checkpoint 的整份副本
    #   · 含中文的 checkpoints 路径换成 ASCII 别名（D:\vc）
    #   · 语义模型（w2v-bert）bf16 加载，省 ~1.1 GB 提交量
    #   · 末尾 20ms 升余弦淡出 + 250ms 静音（修「吞尾音」）
    # 以前 worker 自己写了一份加载代码，上面四条全都拿不到。
    try:
        from indextts_engine import IndexTTSEngine
    except Exception as e:
        _err("import indextts_engine failed", e)
        return 1

    try:
        engine = IndexTTSEngine(use_qwen_emo=False)
    except Exception as e:
        _err("engine init failed", e)
        return 1

    t0 = time.time()
    try:
        load_s = engine.load()
    except Exception as e:
        _err("model load failed", e)
        return 1

    _send({"ready": True, "load_s": round(time.time() - t0, 1),
           "sr": engine.sample_rate})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _err("bad json", e)
            continue

        if req.get("cmd") == "quit":
            break

        rid = req.get("id")
        try:
            text = (req.get("text") or "").strip()
            ref = req.get("ref") or ""
            out = req.get("output") or ""
            if not text:
                raise ValueError("text 为空")
            if not os.path.isfile(ref):
                raise FileNotFoundError(f"参考音频不存在: {ref}")
            if not out:
                raise ValueError("output 为空")

            info = engine.synthesize(
                text,
                ref,
                lang=req.get("lang") or "ZH",
                duration_factor=float(req.get("duration_factor") or 1.0),
                diffusion_steps=int(req.get("diffusion_steps") or DEFAULT_DIFFUSION_STEPS),
                num_beams=int(req.get("num_beams") or 1),
                peak_limit=float(req.get("peak_limit") or 0.95),
                output_path=out,
            )
            _send({
                "id": rid, "ok": True, "output": info["output"],
                "dur": info["dur"], "peak": info["peak"],
                "infer_s": info["infer_s"], "rtf": info["rtf"],
            })
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
