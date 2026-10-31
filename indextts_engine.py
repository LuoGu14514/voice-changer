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


# ── 低内存加载 ──────────────────────────────────────────────────────────
# 这台机器物理内存 15.2 GB，而系统提交量长期在 28 GB 左右。IndexTTS 的
# gpt.pth 有 3.1 GB，torch.load 默认会把整个 checkpoint 反序列化成一份独立
# 副本，再拷进模型 —— 峰值多出 3.1 GB 提交量。这份余量被吃掉之后，后面
# w2v-bert 的 safetensors mmap 物化就拿不到页，torch 直接抛
# "not enough memory"，甚至 ACCESS_VIOLATION (0xC0000005)。
#
# 改成 mmap=True 后，checkpoint 以文件映射方式读入，权重直接拷进模型，
# 省掉整份副本。实测 from_pretrained 由崩溃变成 0.6 s 成功。
_MMAP_ENABLED = False


def _ascii_alias(path: str) -> str:
    """把含中文的路径换成指向同一位置的 ASCII 别名。

    这台机器的项目目录是 D:\\代码\\voice_changer，但它有一个 ASCII 的 junction
    别名 D:\\vc 指向同一位置。torch 的 mmap 加载（load_checkpoint 走
    CreateFileMapping）在中文路径上有已知的踩坑表现：映射建立后指针不可用，
    load_state_dict 里 copy_ 直接 ACCESS_VIOLATION。所以只要能找到同目标的
    ASCII 链接，就改用它。
    """
    if path.isascii():
        return path

    try:
        real = os.path.realpath(path)
    except OSError:
        return path
    real_norm = os.path.normcase(os.path.normpath(real))

    drive = os.path.splitdrive(path)[0] + os.sep
    try:
        entries = list(os.scandir(drive))
    except OSError:
        return path

    for entry in entries:
        try:
            link_abs = os.path.normcase(os.path.abspath(entry.path))
            target = os.path.realpath(entry.path)
            if os.path.normcase(target) == link_abs:
                continue  # 不是链接
        except OSError:
            continue
        target_norm = os.path.normcase(os.path.normpath(target))
        if real_norm == target_norm:
            rel = "."
        elif real_norm.startswith(target_norm + os.sep):
            rel = os.path.relpath(real, target)
        else:
            continue
        alias = entry.path if rel == "." else os.path.join(entry.path, rel)
        if alias.isascii() and os.path.exists(alias):
            return alias
    return path


def _enable_mmap_torch_load(min_size_mb: int = 64) -> None:
    """让 torch.load 对大体积权重文件走 mmap（文件映射）读取。

    只对 .pth / .ckpt 且大于 ``min_size_mb`` 的文件生效；文件不是 zip 格式
    （老式序列化）时 torch 会抛 RuntimeError，此时自动回退到普通加载。
    """
    global _MMAP_ENABLED
    if _MMAP_ENABLED:
        return

    import torch

    original = torch.load
    threshold = min_size_mb * 1024 * 1024

    def patched(f, *args, **kwargs):
        path = f if isinstance(f, (str, os.PathLike)) else None
        use_mmap = False
        if path is not None and str(path).lower().endswith((".pth", ".ckpt")):
            try:
                use_mmap = os.path.getsize(path) >= threshold
            except OSError:
                use_mmap = False
            if use_mmap:
                kwargs.setdefault("mmap", True)
        try:
            return original(f, *args, **kwargs)
        except (RuntimeError, ValueError) as exc:
            if not use_mmap:
                raise
            print(f"[indextts_engine] mmap 不可用（{exc}），回退普通加载", file=sys.stderr)
            kwargs.pop("mmap", None)
            return original(f, *args, **kwargs)

    torch.load = patched
    _MMAP_ENABLED = True


def _release_freed_memory() -> None:
    """把已释放但仍占着物理页的内存还给系统。

    加载过程中 torch 会先后申请和释放 GB 级临时缓冲；不 trim 的话这些页
    仍算作进程占用，压缩提交量会让后续分配失败。
    """
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.windll.psapi.EmptyWorkingSet(-1)
    except Exception:  # noqa: BLE001 - 拿不到就算了，不影响功能
        pass


def _enable_bf16_semantic_model() -> None:
    """让 Wav2Vec2Bert 语义编码器以 bfloat16 加载，省下约 1.3 GB 常驻内存。

    上游只用 use_bf16 转换 GPT / s2mel / bigvgan，w2v-bert 一直是 fp32
    （2.2 GB）。这台机器的提交量本来就吃紧，语义编码器只做特征提取，
    bf16 的精度足够，因此这里把加载精度也降下来。

    直接以 bf16 加载会撞上 dtype 不匹配（下游特征是 fp32），所以还要给
    forward 套一层：输入转 bf16 走算子，输出 hidden_states 转回 fp32。
    加载失败（transformers 版本不接受该参数）时静默回退到 fp32。
    """
    import torch
    from transformers import Wav2Vec2BertModel

    if getattr(Wav2Vec2BertModel, "_vc_bf16_patched", False):
        return

    original = Wav2Vec2BertModel.from_pretrained.__func__

    @classmethod
    def patched(cls, *args, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        try:
            model = original(cls, *args, **kwargs)
        except (TypeError, ValueError, RuntimeError) as exc:
            print(
                f"[indextts_engine] 语义模型 bf16 加载失败（{exc}），回退 fp32",
                file=sys.stderr,
            )
            kwargs.pop("torch_dtype", None)
            return original(cls, *args, **kwargs)

        _patch_semantic_forward(model)
        return model

    Wav2Vec2BertModel.from_pretrained = patched
    Wav2Vec2BertModel._vc_bf16_patched = True


def _patch_semantic_forward(model) -> None:
    """给 bf16 语义模型套一层 dtype 转换：进 bf16、出 fp32。"""
    import torch

    original_forward = model.forward

    def forward(*args, **kwargs):
        if args:
            args = tuple(
                a.to(torch.bfloat16) if torch.is_tensor(a) and a.is_floating_point() else a
                for a in args
            )
        for key in ("input_features", "attention_mask"):
            value = kwargs.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                kwargs[key] = value.to(torch.bfloat16)
        out = original_forward(*args, **kwargs)
        hidden = getattr(out, "hidden_states", None)
        if hidden:
            out.hidden_states = tuple(h.to(torch.float32) for h in hidden)
        return out

    model.forward = forward


class IndexTTSEngine:
    """一个进程内只应存在一个实例；合成必须串行调用。"""

    def __init__(
        self,
        *,
        checkpoints_dir: str | None = None,
        use_qwen_emo: bool = False,
        threads: int | None = None,
        use_bf16: bool = True,
        semantic_bf16: bool = False,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir or CHECKPOINTS
        self.use_qwen_emo = use_qwen_emo
        self.use_bf16 = use_bf16
        self.semantic_bf16 = semantic_bf16
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
        _enable_mmap_torch_load()
        if self.semantic_bf16:
            _enable_bf16_semantic_model()

        import indextts.infer_v2_5 as iv25

        iv25.DIFFUSION_STEPS = DEFAULT_DIFFUSION_STEPS
        iv25.CFG_RATE = DEFAULT_CFG_RATE
        from indextts.infer_v2_5 import IndexTTS2

        cfg_path = os.path.join(self.checkpoints_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"config.yaml 不存在: {cfg_path}")

        # 权重目录换成 ASCII 别名（若有）：torch 的 mmap 在中文路径下会偶发
        # ACCESS_VIOLATION，指针映射建立后不可用。
        load_dir = _ascii_alias(self.checkpoints_dir)
        if load_dir != self.checkpoints_dir:
            print(f"[indextts_engine] 使用 ASCII 别名加载权重: {load_dir}", file=sys.stderr)

        started = time.time()
        self._tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=load_dir,
            use_bf16=self.use_bf16,
            use_cuda_kernel=False,
            use_torch_compile=False,
            use_qwen_emo=self.use_qwen_emo,
        )
        self.load_seconds = round(time.time() - started, 1)
        _release_freed_memory()
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

        # 收尾修复：IndexTTS 的 target_lengths = code_lens * 1.72 由 stop_mel_token
        # 随机采样决定，少数情况下落得偏早，mel 在话音尚未结束时截止，文件最后
        # 一个采样点离零很远 → 听感是"咔"的一声 + 元音尾被吞。补救两步：
        #   1) 末尾 TAIL_FADE_MS 用升余弦渐变到零，消除 click 阶跃
        #   2) 之后追加 TAIL_PAD_MS 静音，让听感有一个自然收尾
        # 上游 utils/common.py 里有同名的 fade_out_tail（20 ms 升余弦），但
        # infer_v2_5.py 走的是 save_pcm_wav()，没有调它——所以这里补上。
        data, sr = sf.read(out, dtype="float64")
        TAIL_FADE_MS = 20.0
        TAIL_PAD_MS = 250.0
        if data.size and sr:
            n_fade = min(int(sr * TAIL_FADE_MS / 1000.0), len(data))
            if n_fade > 1:
                # 升余弦：到零时斜率也为零，淡出本身不引入新的不连续点
                t = np.linspace(0.0, np.pi, n_fade)
                ramp = 0.5 * (1.0 + np.cos(t))
                data[-n_fade:] = data[-n_fade:] * ramp
            n_pad = int(sr * TAIL_PAD_MS / 1000.0)
            if n_pad > 0:
                data = np.concatenate([data, np.zeros(n_pad, dtype=data.dtype)])
            sf.write(out, data, sr, subtype="PCM_16")

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
