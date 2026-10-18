# -*- coding: utf-8 -*-
"""IndexTTS-2.5 引擎封装，供 worker / HTTP server 共用。

把「路径与环境准备」「模型加载」「单次合成」三件事抽出来，避免 worker 和
server 各写一份重复的 bootstrap 代码。

注意：本模块必须用 _indextts_venv 的 Python 3.11 解释器导入，且工作目录应为
ASCII 路径（D:\\vc 这个 junction），因为 wetext 依赖的 kaldifst C++ 层打不开
含中文的路径。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_indextts_src")
CHECKPOINTS = os.path.join(SRC, "checkpoints")

DEFAULT_DIFFUSION_STEPS = 20
DEFAULT_CFG_RATE = 0.7
DEFAULT_PEAK_LIMIT = 0.95

# 支持的语言码（IndexTTS-2.5 是 5 语言模型）
SUPPORTED_LANGS = frozenset({"ZH", "EN", "JA", "ES", "AR"})


def bootstrap_env(threads: int | None = None) -> int:
    """准备 sys.path / 缓存目录 / 线程数。幂等，可重复调用。

    返回实际使用的线程数。
    """
    if SRC not in sys.path:
        sys.path.insert(0, SRC)

    # 缓存全部落到项目内，避开用户主目录的写入限制（ModelScope 的 .lock
    # 曾因为 AppData 不可写而反复报 WinError 5）。
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(CHECKPOINTS, "hf_cache"))
    os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(HERE, "_ms_cache"))
    os.environ.setdefault("USE_MODELSCOPE", "true")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, "_cache", "matplotlib"))
    os.environ.setdefault("HF_HOME", os.path.join(HERE, "_cache", "hf"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(HERE, "_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ("MODELSCOPE_CACHE", "MPLCONFIGDIR", "HF_HOME", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)

    # 默认 torch 只用物理核数（8），这台机器是 8 核 16 线程，用满逻辑核。
    n = int(threads or os.cpu_count() or 8)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    return n


def normalize_lang(value: str | None, fallback: str = "ZH") -> str:
    """把 zh/zh-cn/中文 之类的写法统一成 IndexTTS 认识的两位大写码。"""
    raw = (value or "").strip()
    if not raw:
        return fallback
    alias = {
        "zh-cn": "ZH", "zh-tw": "ZH", "cmn": "ZH", "chinese": "ZH",
        "中文": "ZH", "汉语": "ZH", "普通话": "ZH",
        "en-us": "EN", "en-gb": "EN", "english": "EN", "英语": "EN", "英文": "EN",
        "jp": "JA", "japanese": "JA", "日语": "JA", "日文": "JA",
        "spanish": "ES", "西班牙语": "ES",
        "arabic": "AR", "阿拉伯语": "AR",
    }
    lowered = raw.lower()
    if lowered in alias:
        return alias[lowered]
    upper = raw.upper()
    if upper in SUPPORTED_LANGS:
        return upper
    return upper[:2] if len(upper) >= 2 and upper[:2] in SUPPORTED_LANGS else fallback


class IndexTTSEngine:
    """一个进程内只应存在一个实例；合成必须串行调用。"""

    def __init__(
        self,
        *,
        checkpoints_dir: str | None = None,
        use_qwen_emo: bool = False,
        threads: int | None = None,
        use_bf16: bool = True,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir or CHECKPOINTS
        self.use_qwen_emo = use_qwen_emo
        self.use_bf16 = use_bf16
        self.threads = bootstrap_env(threads)
        self._tts = None
        self.load_seconds = 0.0
        self.sample_rate = 22050

    # ── 加载 ────────────────────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._tts is not None

    def load(self) -> float:
        """加载模型，返回耗时秒数。失败时抛异常。"""
        if self._tts is not None:
            return self.load_seconds

        import torch

        torch.set_num_threads(self.threads)

        import indextts.infer_v2_5 as iv25

        iv25.DIFFUSION_STEPS = DEFAULT_DIFFUSION_STEPS
        iv25.CFG_RATE = DEFAULT_CFG_RATE
        from indextts.infer_v2_5 import IndexTTS2

        cfg_path = os.path.join(self.checkpoints_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"config.yaml 不存在: {cfg_path}")

        started = time.time()
        self._tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=self.checkpoints_dir,
            use_bf16=self.use_bf16,
            use_cuda_kernel=False,
            use_torch_compile=False,
            use_qwen_emo=self.use_qwen_emo,
        )
        self.load_seconds = round(time.time() - started, 1)
        try:
            self.sample_rate = int(getattr(self._tts, "sr", 22050) or 22050)
        except Exception:  # noqa: BLE001 - 取不到就沿用默认
            self.sample_rate = 22050
        return self.load_seconds

    # ── 合成 ────────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        ref_audio: str,
        *,
        lang: str = "ZH",
        duration_factor: float = 1.0,
        diffusion_steps: int | None = None,
        num_beams: int = 1,
        peak_limit: float = DEFAULT_PEAK_LIMIT,
        output_path: str | None = None,
        with_bytes: bool = False,
    ) -> dict:
        """合成一段语音。

        ``duration_factor`` > 1 更慢更长，< 1 更快更短。
        ``output_path`` 为空时写临时文件。
        返回 dict：output / dur / peak / infer_s / rtf / sr，``with_bytes`` 时
        额外带 ``wav_bytes``。
        """
        if self._tts is None:
            raise RuntimeError("模型尚未加载，先调用 load()")

        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("text 为空")
        if not ref_audio or not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")

        import numpy as np
        import soundfile as sf

        import indextts.infer_v2_5 as iv25

        steps = int(diffusion_steps or DEFAULT_DIFFUSION_STEPS)
        iv25.DIFFUSION_STEPS = steps

        out = output_path
        if not out:
            out = os.path.join(
                tempfile.gettempdir(), f"indextts_{uuid.uuid4().hex}.wav"
            )
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

        started = time.time()
        self._tts.infer(
            spk_audio_prompt=ref_audio,
            text=clean_text,
            lang=normalize_lang(lang),
            output_path=out,
            duration_factor=float(duration_factor or 1.0),
            num_beams=int(num_beams or 1),
            verbose=False,
        )
        infer_s = time.time() - started

        # 峰值保护：不同参考音频的响度差异会让输出顶到 1.0 造成削波。
        data, sr = sf.read(out, dtype="float64")
        peak = float(np.abs(data).max()) if data.size else 0.0
        limit = float(peak_limit or DEFAULT_PEAK_LIMIT)
        if peak > limit > 0:
            sf.write(out, data * (limit / peak), sr, subtype="PCM_16")
            data, sr = sf.read(out, dtype="float64")
            peak = float(np.abs(data).max()) if data.size else limit

        dur = float(len(data) / sr) if sr else 0.0
        result = {
            "output": out,
            "sr": int(sr or self.sample_rate),
            "dur": round(dur, 3),
            "peak": round(peak, 4),
            "infer_s": round(infer_s, 2),
            "rtf": round(infer_s / dur, 2) if dur else None,
            "diffusion_steps": steps,
        }
        if with_bytes:
            with open(out, "rb") as handle:
                result["wav_bytes"] = handle.read()
        return result


def describe_exception(exc: BaseException) -> str:
    """统一错误描述格式。"""
    return f"{type(exc).__name__}: {exc}"


def log_exception(exc: BaseException) -> str:
    """把 traceback 打到 stderr 并返回一行描述。"""
    traceback.print_exc(file=sys.stderr)
    return describe_exception(exc)
