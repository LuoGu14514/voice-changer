# -*- coding: utf-8 -*-
"""简单变声器 —— DSP 核心。

只做两件事：
1. StreamingPitchShifter：实时变调（颗粒式重叠相加 + WSOLA 相位对齐，
   保持时长不变），可以在音频回调里逐块调用，音调比值可随时改变。
2. EffectChain：机器人 / 回声 / 电话音等简单音效。

不依赖 scipy，只需要 numpy。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "semitones_to_ratio",
    "StreamingPitchShifter",
    "EffectChain",
    "process_offline",
    "EFFECTS",
    "EFFECT_LABELS",
]

EFFECTS = ("none", "robot", "echo", "phone")
EFFECT_LABELS = {"none": "无", "robot": "机器人", "echo": "回声", "phone": "电话音"}


def semitones_to_ratio(semitones: float) -> float:
    """半音数 -> 频率比值。+12 半音 = 2.0（升高一个八度）。"""
    return float(2.0 ** (float(semitones) / 12.0))


def _hann(n: int) -> np.ndarray:
    """周期 Hann 窗。长度 n、相邻窗错开 n/2 时，两窗之和恒等于 1。"""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


class StreamingPitchShifter:
    """实时变调器（颗粒合成 / 重叠相加 + WSOLA 对齐）。

    原理：每 B 个采样输出一颗长度为 W = 2B 的颗粒，颗粒内部以 ratio 倍速度
    读取输入（这一步就是变调），相邻颗粒起点的输入位置只差 B（所以总时长
    不变），两颗颗粒用 Hann 窗交叉淡化，窗和为 1 所以音量不变。

    只做上面这些的话，相邻颗粒在交叉处相位是断的，频谱会碎成一堆间隔
    sr/B 的梳状谱线 —— 音高听着发飘、发哑。所以这里再加一步很便宜的
    WSOLA 对齐：在新颗粒起点附近 ±search 个采样里，挑一个和上一颗颗粒
    尾巴（正在淡出的那段）相关度最高的位置，让波形接得上。对纯音而言
    这样拼出来的就是干净的变调正弦。

    每次 process() 处理一块 B 个采样，块内完全向量化，可以直接放进
    sounddevice 的回调里。除了对齐搜索那几十万个乘加没有别的开销。

    latency_samples 是算法本身的延迟（与音调无关，恒定）。
    """

    def __init__(self, block_size: int = 512, max_ratio: float = 2.0,
                 align: bool = True, search_ratio: float = 0.25):
        self.B = int(block_size)
        self.W = 2 * self.B
        self.max_ratio = float(max_ratio)
        self.align = bool(align)
        self.win = _hann(self.W)
        self.search = max(4, int(round(self.B * float(search_ratio))))
        # 对齐搜索的候选偏移（大步长时按比例抽样，保证候选数在 257 左右）
        step = max(1, self.search // 128)
        self._cands = np.arange(-self.search, self.search + 1, step, dtype=np.int64)
        stride = max(1, self.B // 256)
        self._corr_idx = np.arange(0, self.B, stride)
        # 最坏情况（升 12 半音 ratio=2，且对齐偏移取到最远处）需要的 look-ahead
        self.span = int(np.ceil(self.max_ratio * self.W)) + self.search + 8
        self.ring_len = 1 << max(4, int(np.ceil(np.log2(
            self.span + self.W + self.search + 4 * self.B + 32))))
        self.reset()

    # ---------------------------------------------------------------- 状态
    def reset(self) -> None:
        self.ring = np.zeros(self.ring_len, dtype=np.float32)
        self.w = 0          # 累计写入的输入采样数（绝对位置）
        # 颗粒起点。起始为负：表示"这些采样还没录进来"，读到时按静音处理，
        # 于是开头自带一段延迟而不是咔哒声。
        self.s = self.B - self.span
        self.acc = np.zeros(self.B, dtype=np.float32)   # 上一颗颗粒的后半段
        self.has_acc = False
        self.ratio = 1.0
        self.read_overflow = 0   # 诊断用：读到还没写入的采样就 +1（应为 0）

    @property
    def latency_samples(self) -> int:
        """输出比输入慢多少个采样。

        推导：输出第 j 个采样由第 k = j // B 颗颗粒贡献，其起点是
        s_k = (k+1)B - span，于是 out[j] = in[j - (span - B)]。
        """
        return self.span - self.B

    # ---------------------------------------------------------------- 内部
    def _best_offset(self, s: float, ratio: float) -> int:
        """在 ±search 里找一个让新颗粒和上一颗颗粒尾巴最接得上的起点偏移。"""
        B, R = self.B, self.ring_len
        n = self._corr_idx.astype(np.float64)
        cands = self._cands
        pos = s + cands[:, None].astype(np.float64) + ratio * n[None, :]
        i0 = np.floor(pos).astype(np.int64)
        frac = (pos - i0).astype(np.float32)
        i0 %= R
        i1 = (i0 + 1) % R
        cur = self.ring[i0] * (1.0 - frac) + self.ring[i1] * frac
        cur *= self.win[self._corr_idx]
        cur[pos < 0.0] = 0.0
        den = np.sqrt(np.einsum("ij,ij->i", cur, cur)) + 1e-9
        score = (cur @ self.acc[self._corr_idx]) / den
        # 轻微偏向 0 偏移：周期信号会有多个同样好的解，别乱跳
        score -= 1e-3 * (cands / max(1, self.search)) ** 2
        return int(cands[int(np.argmax(score))])

    # ---------------------------------------------------------------- 处理
    def process(self, x, ratio: float) -> np.ndarray:
        """输入一块（长度 B）采样，返回同长度的变调结果。"""
        B, W, R = self.B, self.W, self.ring_len
        self.ratio = float(ratio)

        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != B:
            buf = np.zeros(B, dtype=np.float32)
            n = min(B, x.size)
            buf[:n] = x[:n]
            x = buf

        # 1) 写入环形缓冲（顺序写，位置就是 w % R）
        p = self.w % R
        end = p + B
        if end <= R:
            self.ring[p:end] = x
        else:
            k = R - p
            self.ring[p:] = x[:k]
            self.ring[: B - k] = x[k:]
        self.w += B

        # 2) 选颗粒起点（对齐搜索；第一颗没有尾巴可对，就用名义位置）
        #    音调不变时（ratio=1）必须严格原样通过：重叠相加本身已经精确，
        #    再去做对齐搜索反而会把瞬态（爆破音、鼓点）在时间轴上挪来挪去。
        s = self.s
        if self.align and self.has_acc and abs(self.ratio - 1.0) > 1e-6:
            off = self._best_offset(s, self.ratio)
        else:
            off = 0

        # 3) 读一颗颗粒：位置 = s + off + ratio * n，线性插值。
        #    pos 可能为负（开头还没录到的地方），这些位置按静音处理，
        #    所以必须用 floor 而不是截断，否则负位置的插值权重会变成负的。
        n = np.arange(W, dtype=np.float64)
        pos = (s + off) + self.ratio * n
        if pos[-1] >= self.w:
            self.read_overflow += 1
        i0 = np.floor(pos).astype(np.int64)
        frac = (pos - i0).astype(np.float32)
        i0 %= R
        i1 = (i0 + 1) % R
        g = self.ring[i0] * (1.0 - frac) + self.ring[i1] * frac
        g *= self.win
        g[pos < 0.0] = 0.0

        # 4) 与上一颗颗粒的后半段相加 -> 输出前半段
        out = self.acc + g[:B]
        self.acc = g[B:]
        self.has_acc = True
        self.s = s + B          # 名义起点每次只前进 B，保证平均速率 = 1（不漂移）
        return out.astype(np.float32)


class EffectChain:
    """几种便宜的变声音效，状态跨块保留。"""

    def __init__(self, samplerate: int = 48000, block_size: int = 512,
                 echo_ms: float = 130.0, echo_feedback: float = 0.35,
                 echo_mix: float = 0.45, ringmod_hz: float = 60.0,
                 phone_lo: float = 400.0, phone_hi: float = 3000.0, phone_taps: int = 201):
        self.sr = int(samplerate)
        self.B = int(block_size)
        self.mode = "none"
        self.echo_ms = float(echo_ms)
        self.echo_feedback = float(echo_feedback)
        self.echo_mix = float(echo_mix)
        self.ringmod_hz = float(ringmod_hz)
        self.phone_lo, self.phone_hi = float(phone_lo), float(phone_hi)
        self.phone_taps = int(phone_taps)
        self._h = self._design_bandpass()
        self.reset()

    # ---------------------------------------------------------------- 内部
    def _design_bandpass(self) -> np.ndarray:
        """加窗 sinc 带通 FIR，通带增益归一化到 1。"""
        taps = self.phone_taps
        n = np.arange(taps) - (taps - 1) / 2.0
        f1, f2 = self.phone_lo / self.sr, self.phone_hi / self.sr
        h = 2.0 * f2 * np.sinc(2.0 * f2 * n) - 2.0 * f1 * np.sinc(2.0 * f1 * n)
        h *= np.hamming(taps)
        f0 = np.sqrt(self.phone_lo * self.phone_hi)
        w = np.exp(-2j * np.pi * f0 / self.sr * np.arange(taps))
        gain = abs(np.dot(h, w))
        if gain > 0:
            h = h / gain
        return h.astype(np.float32)

    def reset(self) -> None:
        self._phase = 0.0
        d = max(1, int(self.sr * self.echo_ms / 1000.0))
        self._delay = np.zeros(d, dtype=np.float32)
        self._dp = 0
        self._tail = np.zeros(max(0, self.phone_taps - 1), dtype=np.float32)

    def set_mode(self, mode: str) -> None:
        mode = mode if mode in EFFECTS else "none"
        if mode != self.mode:
            self.mode = mode
            self.reset()

    # ---------------------------------------------------------------- 处理
    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        m = self.mode
        if m == "none" or x.size == 0:
            return x
        if m == "robot":
            inc = 2.0 * np.pi * self.ringmod_hz / self.sr
            ph = self._phase + inc * np.arange(x.size)
            self._phase = float((self._phase + inc * x.size) % (2.0 * np.pi))
            return (x * np.sin(ph)).astype(np.float32)
        if m == "echo":
            d = self._delay.size
            out = np.empty_like(x)
            for i in range(x.size):
                v = self._delay[self._dp]
                out[i] = x[i] + self.echo_mix * v
                self._delay[self._dp] = np.float32(x[i] + self.echo_feedback * v)
                self._dp += 1
                if self._dp >= d:
                    self._dp = 0
            return out
        if m == "phone":
            full = np.convolve(x, self._h)
            out = full[: x.size].copy()
            if self._tail.size:
                out[: self._tail.size] += self._tail
            self._tail = full[x.size:].astype(np.float32)
            # 一点点过载，更像电话/对讲机
            return (np.tanh(out * 1.6) * 0.9).astype(np.float32)
        return x


def process_offline(x, samplerate: int, semitones: float = 0.0, effect: str = "none",
                    block_size: int = 512, gain: float = 1.0,
                    align: bool = True) -> np.ndarray:
    """离线处理一整段音频（文件变声用）。输出长度与输入一致。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    B = int(block_size)
    shifter = StreamingPitchShifter(B, align=align)
    effects = EffectChain(samplerate, B)
    effects.set_mode(effect)
    ratio = semitones_to_ratio(semitones)
    delay = shifter.latency_samples

    # 输出比输入慢 delay 个采样，所以要多喂 delay 个采样才够裁回来
    total = int(x.size) + int(delay)
    xp_size = ((total + B - 1) // B) * B
    xp = np.concatenate([x, np.zeros(xp_size - x.size, dtype=np.float32)])

    chunks = []
    for i in range(0, xp.size, B):
        y = shifter.process(xp[i:i + B], ratio)
        chunks.append(effects.process(y))
    y = np.concatenate(chunks) * float(gain)
    y = y[delay: delay + x.size]
    if y.size < x.size:
        y = np.concatenate([y, np.zeros(x.size - y.size, dtype=np.float32)])
    return y.astype(np.float32)
