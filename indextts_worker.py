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
    # CPU 线程数：默认只有物理核数，这里用满逻辑核
    nthreads = os.cpu_count() or 8
    os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))
    os.environ.setdefault("MKL_NUM_THREADS", str(nthreads))

    try:
        import numpy as np
        import soundfile as sf
        import torch

        torch.set_num_threads(nthreads)

        import indextts.infer_v2_5 as iv25
        iv25.DIFFUSION_STEPS = DEFAULT_DIFFUSION_STEPS
        iv25.CFG_RATE = DEFAULT_CFG_RATE
        from indextts.infer_v2_5 import IndexTTS2
    except Exception as e:
        _err("import failed", e)
        return 1

    cfg = os.path.join(CHECKPOINTS, "config.yaml")
    if not os.path.isfile(cfg):
        _err(f"config.yaml 不存在: {cfg}")
        return 1

    t0 = time.time()
    try:
        tts = IndexTTS2(
            cfg_path=cfg,
            model_dir=CHECKPOINTS,
            use_bf16=True,
            use_cuda_kernel=False,
            use_torch_compile=False,
            use_qwen_emo=False,   # GUI 不做情感文本推断，省 1.19 GB 内存
        )
    except Exception as e:
        _err("model load failed", e)
        return 1

    _send({"ready": True, "load_s": round(time.time() - t0, 1), "sr": 22050})

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

            steps = int(req.get("diffusion_steps") or DEFAULT_DIFFUSION_STEPS)
            iv25.DIFFUSION_STEPS = steps

            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

            t1 = time.time()
            tts.infer(
                spk_audio_prompt=ref,
                text=text,
                lang=req.get("lang") or "ZH",
                output_path=out,
                duration_factor=float(req.get("duration_factor") or 1.0),
                num_beams=int(req.get("num_beams") or 1),
                verbose=False,
            )
            infer_s = time.time() - t1

            # 峰值保护（不同参考音频响度差异会让输出顶到 1.0）
            data, sr = sf.read(out, dtype="float64")
            peak = float(np.abs(data).max()) if data.size else 0.0
            limit = float(req.get("peak_limit") or 0.95)
            if peak > limit and peak > 0:
                data = data * (limit / peak)
                sf.write(out, data, sr, subtype="PCM_16")
                peak = limit
                data, sr = sf.read(out)

            dur = len(data) / sr if sr else 0.0
            _send({
                "id": rid, "ok": True, "output": out,
                "dur": round(dur, 2), "peak": round(peak, 3),
                "infer_s": round(infer_s, 1),
                "rtf": round(infer_s / dur, 2) if dur else None,
            })
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
