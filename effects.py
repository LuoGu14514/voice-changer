# -*- coding: utf-8 -*-
"""12+ 效果器 rack —— 借鉴 sioaeko/OpenVoiceChanger。

每个效果：__init__(sr, B) + reset() + process(x, **params)。
EffectRack 维护所有效果实例，按 config 顺序串联处理。
所有效果纯 numpy、零依赖、状态跨块保留，能放进音频回调。

实现策略：所有循环都尽量向量化（一次处理整块 B 个采样）。做不到的就
退到逐采样（只对极便宜的效果，比如 envelope follower）。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "EFFECT_NAMES",
    "EFFECT_LABELS",
    "NoiseGate",
    "Robot",
    "Whisper",
    "Telephone",
    "Distortion",
    "Bitcrush",
    "Chorus",
    "Echo",
    "Reverb",
    "ToneEQ",
    "Compressor",
    "OutputGain",
    "BreathNoise",
    "Tremolo",
    "Vibrato",
    "SilenceSaver",
    "EffectRack",
    "build_effect_rack",
]


EFFECT_NAMES = (
    "noise_gate", "robot", "whisper", "telephone", "distortion", "bitcrush",
    "chorus", "echo", "reverb", "tone_eq", "compressor", "output_gain",
    "tremolo", "vibrato", "breath",
)
EFFECT_LABELS = {
    "noise_gate":  "降噪门",
    "robot":       "机器人",
    "whisper":     "气音",
    "telephone":   "电话音",
    "distortion":  "失真",
    "bitcrush":    "位破坏",
    "chorus":      "合唱",
    "echo":        "回声",
    "reverb":      "混响",
    "tone_eq":     "音色 EQ",
    "compressor":  "压缩器",
    "output_gain": "输出增益",
    "tremolo":     "颤音",
    "vibrato":     "震音",
    "breath":      "呼吸声",
}


# =============================================================================
# 基础一阶 IIR（lowpass / highpass）—— 全程向量化
# =============================================================================
def _iir_lp(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """一阶低通，零依赖、纯 numpy。"""
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))
    y = np.empty_like(x)
    z = np.float32(0.0)
    # 这里只能循环（一阶 IIR 必须保留前一输出），但每次只有 1 个乘 + 1 个加，
    # B=512 时 ~ 0.1ms。
    for i in range(x.size):
        z = (1.0 - a) * x[i] + a * z
        y[i] = z
    return y


def _iir_hp(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """一阶高通：x - LP(x)。"""
    lp = _iir_lp(x, cutoff_hz, sr)
    return (x - lp).astype(np.float32)


# =============================================================================
# 1. NoiseGate —— 降噪门
# =============================================================================
class NoiseGate:
    """RMS 阈值门：低于阈值时指数衰减到 0；高于时打开。

    attack_ms：门打开的速度；release_ms：门关闭的速度。
    threshold_db：低于此 RMS 的输入视为静音。
    """

    def __init__(self, sr: int, B: int,
                 threshold_db: float = -50.0,
                 attack_ms: float = 5.0, release_ms: float = 50.0):
        self.sr = int(sr); self.B = int(B)
        self.threshold = 10 ** (float(threshold_db) / 20.0)
        self.alpha_a = float(np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0)))
        self.alpha_r = float(np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0)))
        self._gain = 0.0

    def reset(self):
        self._gain = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        # RMS 包络：用 cumsum 算滑窗 RMS
        win = max(16, self.sr * 5 // 1000)
        if x.size < win:
            env = np.full(x.size, float(np.sqrt(np.mean(x * x) + 1e-12)), dtype=np.float32)
        else:
            sq = x * x
            cs = np.cumsum(sq, dtype=np.float32)
            cs[win:] = cs[win:] - cs[:-win]
            env = np.sqrt(cs[win - 1:] / win + 1e-12).astype(np.float32)
            env = np.concatenate([np.full(x.size - env.size, env[0], dtype=np.float32), env])
        # 每采样 gain 包络（向量化：先分段，再展开）
        target = (env > self.threshold).astype(np.float32)
        # 一阶 IIR gain 包络 —— 不得不循环（B=512 单次 ~ 0.05ms 可接受）
        g = np.empty_like(x)
        gain = np.float32(self._gain)
        a_a = np.float32(self.alpha_a); a_r = np.float32(self.alpha_r)
        for i in range(x.size):
            t = target[i]
            if t > gain:
                gain = t + (gain - t) * a_a
            else:
                gain = t + (gain - t) * a_r
            g[i] = gain
        self._gain = float(gain)
        return (x * g).astype(np.float32)


# =============================================================================
# 2. Robot —— 环形调制
# =============================================================================
class Robot:
    """x * sin(2π·hz·t)。hz 在 ~50-300Hz 时像机器人声，更高是颤/震音。"""

    def __init__(self, sr: int, B: int, hz: float = 60.0):
        self.sr = int(sr); self.B = int(B)
        self.hz = float(hz)
        self._phase = 0.0

    def reset(self):
        self._phase = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        inc = 2.0 * np.pi * self.hz / self.sr
        n = x.size
        ph = self._phase + inc * np.arange(n)
        self._phase = float((self._phase + inc * n) % (2.0 * np.pi))
        return (x * np.sin(ph).astype(np.float32)).astype(np.float32)


# =============================================================================
# 3. Whisper —— 气音（破坏谐波 + 抬高频噪声）
# =============================================================================
class Whisper:
    """把信号包络保留，细节替换为高通噪声 —— 像在耳边吹气说话。

    strength 0~1：0 = 原样，1 = 几乎纯气声。
    """

    def __init__(self, sr: int, B: int, hp_cut: float = 1500.0):
        self.sr = int(sr); self.B = int(B)
        self.hp_cut = float(hp_cut)
        self._rng = np.random.default_rng()

    def reset(self):
        pass

    def process(self, x: np.ndarray, strength: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        s = float(np.clip(strength, 0.0, 1.0))
        n = x.size
        env = np.abs(x)
        noise = self._rng.standard_normal(n).astype(np.float32) * env * 0.6
        hp = _iir_hp(noise, self.hp_cut, self.sr)
        return ((1.0 - s) * x + s * hp).astype(np.float32)


# =============================================================================
# 4. Telephone —— 电话音（300-3400Hz 带通 + tanh 软削）
# =============================================================================
class Telephone:
    """电话音：300-3400Hz 带通 + 1.6× tanh 过载。"""

    def __init__(self, sr: int, B: int,
                 lo: float = 300.0, hi: float = 3400.0, taps: int = 201):
        self.sr = int(sr); self.B = int(B)
        self.taps = max(31, int(taps) | 1)
        self.lo = float(lo); self.hi = float(hi)
        self._h = self._design()
        self._tail = np.zeros(self.taps - 1, dtype=np.float32)

    def reset(self):
        self._tail.fill(0)

    def _design(self) -> np.ndarray:
        n = np.arange(self.taps) - (self.taps - 1) / 2.0
        f1, f2 = self.lo / self.sr, self.hi / self.sr
        h = 2.0 * f2 * np.sinc(2.0 * f2 * n) - 2.0 * f1 * np.sinc(2.0 * f1 * n)
        h *= np.hamming(self.taps)
        f0 = np.sqrt(self.lo * self.hi)
        w = np.exp(-2j * np.pi * f0 / self.sr * np.arange(self.taps))
        g = abs(np.dot(h, w))
        if g > 0:
            h = h / g
        return h.astype(np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        full = np.convolve(x, self._h, mode="full")
        out = full[: x.size].copy()
        if self._tail.size:
            out[: self._tail.size] += self._tail
        self._tail = full[x.size:].astype(np.float32)
        return (np.tanh(out * 1.6) * 0.9).astype(np.float32)


# =============================================================================
# 5. Distortion —— 软失真
# =============================================================================
class Distortion:
    """tanh(drive * x) / tanh(drive)。drive=1 几乎线性，drive=10 重失真。"""

    def __init__(self, sr: int, B: int):
        self.sr = int(sr); self.B = int(B)

    def reset(self):
        pass

    def process(self, x: np.ndarray, drive: float = 2.0, mix: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        d = float(np.clip(drive, 0.5, 20.0))
        m = float(np.clip(mix, 0.0, 1.0))
        norm = float(np.tanh(d))
        wet = np.tanh(d * x) / norm
        return ((1.0 - m) * x + m * wet).astype(np.float32)


# =============================================================================
# 6. Bitcrush —— 位破坏
# =============================================================================
class Bitcrush:
    """位破坏：bits=16→4 + downsample=1→16。

    strength 0~1：0 = 原样，1 = 最狠（4 bit + 16× 降采样）。
    """

    def __init__(self, sr: int, B: int):
        self.sr = int(sr); self.B = int(B)

    def reset(self):
        pass

    def process(self, x: np.ndarray, strength: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        s = float(np.clip(strength, 0.0, 1.0))
        bits = int(16 - 12 * s)
        bits = max(2, bits)
        ds = max(1, int(1 + 15 * s))
        if bits >= 16 and ds == 1:
            return x
        levels = float(2 ** bits)
        y = np.round(x * levels) / levels
        if ds > 1:
            n = y.size
            # 每 ds 个采样 hold 一次：取每段的第 0 个采样，再 repeat ds 次
            n_held = n // ds
            held = y[:n_held * ds:ds]  # n_held 个采样
            y2 = np.repeat(held, ds)   # n_held * ds 个采样
            if y2.size < n:
                y2 = np.concatenate([y2, np.full(n - y2.size, y2[-1], dtype=np.float32)])
            y = y2
        return y.astype(np.float32)


# =============================================================================
# 7. Chorus —— 3 路合唱
# =============================================================================
class Chorus:
    """3 路不同延迟 + LFO 调制。"""

    def __init__(self, sr: int, B: int):
        self.sr = int(sr); self.B = int(B)
        self.base_ms = np.array([15.0, 22.0, 29.0], dtype=np.float32)
        self.lfo_hz = np.array([0.7, 1.1, 1.7], dtype=np.float32)
        self.depth_ms = 4.0
        max_d_samp = int(sr * (float(self.base_ms.max()) + self.depth_ms + 4.0) / 1000.0) + 8
        self._buf_size = max_d_samp + B + 64
        self._buf = np.zeros((3, self._buf_size), dtype=np.float32)
        self._wptr = 0

    def reset(self):
        self._buf.fill(0)
        self._wptr = 0

    def process(self, x: np.ndarray, mix: float = 0.4) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        m = float(np.clip(mix, 0.0, 1.0))
        n = x.size
        L = self._buf_size
        wptr = self._wptr
        out = np.zeros(n, dtype=np.float32)
        # 3 路延迟
        for k in range(3):
            base = self.base_ms[k] * 0.001 * self.sr  # 延迟采样数
            depth = self.depth_ms * 0.001 * self.sr
            lfo_hz = self.lfo_hz[k]
            # 读位置（相对）：r[i] = i - (base + depth * sin(...))
            t = np.arange(n, dtype=np.float32)
            lfo = depth * np.sin(2.0 * np.pi * lfo_hz / self.sr * t + k * 0.5)
            r_f = (wptr + t) - (base + lfo)
            # 环形插值
            r_f_mod = np.mod(r_f, L)
            r0 = np.floor(r_f_mod).astype(np.int64)
            r1 = (r0 + 1) % L
            frac = (r_f_mod - r0).astype(np.float32)
            v = self._buf[k, r0] * (1.0 - frac) + self._buf[k, r1] * frac
            out += v / 3.0
        # 写入 x 到环形缓冲
        wp_idx = (wptr + np.arange(n)) % L
        for k in range(3):
            self._buf[k, wp_idx] = x
        self._wptr = (wptr + n) % L
        return ((1.0 - m) * x + m * out).astype(np.float32)


# =============================================================================
# 8. Echo —— 延迟反馈
# =============================================================================
class Echo:
    """y[n] = x[n] + mix * y[n - D]，buf 更新：buf[p] = x[n] + fb * buf[p]。
    一次 IIR 反馈 + 一次 dry/wet 混合。
    """

    def __init__(self, sr: int, B: int,
                 delay_ms: float = 130.0, feedback: float = 0.35, mix: float = 0.45):
        self.sr = int(sr); self.B = int(B)
        self.delay_ms = float(delay_ms)
        self.feedback = float(np.clip(feedback, 0.0, 0.95))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self._d = max(1, int(sr * self.delay_ms / 1000.0))
        self._buf = np.zeros(self._d, dtype=np.float32)
        self._p = 0

    def reset(self):
        self._buf.fill(0); self._p = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        d = self._d; fb = self.feedback; mix = self.mix
        n = x.size
        out = np.empty(n, dtype=np.float32)
        buf = self._buf; p = self._p
        for i in range(n):
            v = buf[p]
            out[i] = x[i] + mix * v
            buf[p] = np.float32(x[i] + fb * v)
            p += 1
            if p >= d: p = 0
        self._p = p
        return out.astype(np.float32)


# =============================================================================
# 9. Reverb —— Schroeder 混响（4 comb + 2 allpass）
# =============================================================================
class Reverb:
    """经典 Schroeder 混响：4 comb 并联求和 → 2 allpass 串联 → 与 dry 混合。"""

    def __init__(self, sr: int, B: int):
        self.sr = int(sr); self.B = int(B)
        self.comb_ms = [29.7, 37.1, 41.1, 43.7]
        self.comb_g = 0.84
        self.ap_ms = [5.0, 1.7]
        self.ap_g = 0.7
        self._comb = [np.zeros(max(1, int(sr * ms / 1000.0)), dtype=np.float32)
                      for ms in self.comb_ms]
        self._comb_p = [0] * len(self._comb)
        self._ap = [np.zeros(max(1, int(sr * ms / 1000.0)), dtype=np.float32)
                    for ms in self.ap_ms]
        self._ap_p = [0] * len(self._ap)

    def reset(self):
        for b in self._comb: b.fill(0)
        for b in self._ap: b.fill(0)
        self._comb_p = [0] * len(self._comb)
        self._ap_p = [0] * len(self._ap)

    def process(self, x: np.ndarray, mix: float = 0.3, room: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        m = float(np.clip(mix, 0.0, 1.0))
        g = float(np.clip(self.comb_g * room, 0.0, 0.95))
        n = x.size
        # 4 comb 并联
        wet = np.zeros(n, dtype=np.float32)
        for k in range(len(self._comb)):
            buf = self._comb[k]; d = buf.size; p = self._comb_p[k]
            out_k = np.empty(n, dtype=np.float32)
            for i in range(n):
                v = buf[p]
                out_k[i] = v
                buf[p] = np.float32(x[i] + g * v)
                p += 1
                if p >= d: p = 0
            self._comb_p[k] = p
            wet += out_k
        wet *= 0.25
        # 2 allpass 串联
        for k in range(len(self._ap)):
            buf = self._ap[k]; d = buf.size; p = self._ap_p[k]
            ag = self.ap_g
            for i in range(n):
                v = buf[p]
                buf[p] = np.float32(wet[i] + ag * v)
                wet[i] = np.float32(-ag * wet[i] + v)
                p += 1
                if p >= d: p = 0
            self._ap_p[k] = p
        return ((1.0 - m) * x + m * wet).astype(np.float32)


# =============================================================================
# 10. ToneEQ —— 3 段 EQ（bass / mid peak / treble）
# =============================================================================
class ToneEQ:
    """3 段 EQ，频域实现（每块 1 次 rfft + 1 次 irfft）。

    bass_db / mid_db / treble_db：每段增益（dB）。
    bass_freq=200Hz, mid_freq=1000Hz Q=1.0, treble_freq=4000Hz。
    """

    def __init__(self, sr: int, B: int,
                 bass_freq: float = 200.0, mid_freq: float = 1000.0,
                 mid_q: float = 1.0, treble_freq: float = 4000.0):
        self.sr = int(sr); self.B = int(B)
        self.bass_freq = float(bass_freq)
        self.mid_freq = float(mid_freq)
        self.mid_q = float(mid_q)
        self.treble_freq = float(treble_freq)
        # 预计算 N（最小 2 的幂 ≥ 2B），保证 freq grid 稳定
        self._N = 1 << int(np.ceil(np.log2(max(B * 2, 64))))
        self._f = np.fft.rfftfreq(self._N, 1.0 / self.sr).astype(np.float32)

    def reset(self):
        pass

    def process(self, x: np.ndarray,
                bass_db: float = 0.0, mid_db: float = 0.0, treble_db: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        if abs(bass_db) < 0.01 and abs(mid_db) < 0.01 and abs(treble_db) < 0.01:
            return x
        n = x.size
        X = np.fft.rfft(x, self._N)
        H = np.ones_like(X, dtype=np.float32)
        f = self._f
        # 低架（200Hz）：f→0 增益=A，f→∞ 增益=1
        if abs(bass_db) > 0.01:
            A = 10 ** (bass_db / 20.0)
            shelf_low = 1.0 / (1.0 + (f / self.bass_freq) ** 2) ** 0.5  # 1→0
            H *= 1.0 + (A - 1.0) * shelf_low
        # 中峰（1000Hz, Q=1.0）：高斯形状
        if abs(mid_db) > 0.01:
            bw = self.mid_freq / self.mid_q
            d = (f - self.mid_freq) / (bw / 2.0)
            H *= (10 ** (mid_db / 20.0)) ** np.exp(-d * d * 0.5)
        # 高架（4000Hz）：f→0 增益=1，f→∞ 增益=A
        if abs(treble_db) > 0.01:
            A = 10 ** (treble_db / 20.0)
            shelf_high = (f / self.treble_freq) / (1.0 + (f / self.treble_freq) ** 2) ** 0.5  # 0→1
            H *= 1.0 + (A - 1.0) * shelf_high
        y = np.fft.irfft(X * H.astype(np.complex64), self._N).real[:n]
        return y.astype(np.float32)


# =============================================================================
# 11. Compressor —— 简单压缩器
# =============================================================================
class Compressor:
    """包络压缩：threshold_db + ratio + attack_ms + release_ms + makeup_db。"""

    def __init__(self, sr: int, B: int,
                 threshold_db: float = -20.0, ratio: float = 3.0,
                 attack_ms: float = 5.0, release_ms: float = 80.0,
                 makeup_db: float = 0.0):
        self.sr = int(sr); self.B = int(B)
        self.threshold = 10 ** (threshold_db / 20.0)
        self.ratio = float(ratio)
        self.alpha_a = float(np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0)))
        self.alpha_r = float(np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0)))
        self.makeup = 10 ** (makeup_db / 20.0)
        self._env = 0.0

    def reset(self):
        self._env = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        env = self._env; a_a = self.alpha_a; a_r = self.alpha_r
        thr = self.threshold; ratio = self.ratio; makeup = self.makeup
        out = np.empty_like(x)
        for i in range(x.size):
            v = abs(x[i])
            if v > env:
                env = v + (env - v) * a_a
            else:
                env = v + (env - v) * a_r
            if env > thr and env > 1e-9:
                over_db = 20.0 * np.log10(env / thr)
                gr_db = over_db * (1.0 - 1.0 / ratio)
                gr = 10 ** (-gr_db / 20.0)
            else:
                gr = 1.0
            out[i] = x[i] * gr * makeup
        self._env = env
        return out.astype(np.float32)


# =============================================================================
# 12. OutputGain —— 增益
# =============================================================================
class OutputGain:
    """线性增益（dB）。"""

    def __init__(self, sr: int, B: int):
        self.sr = int(sr); self.B = int(B)

    def reset(self):
        pass

    def process(self, x: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        g = 10 ** (float(gain_db) / 20.0)
        return (x * g).astype(np.float32)


# =============================================================================
# 13. Tremolo —— 颤音
# =============================================================================
class Tremolo:
    """振幅按 LFO 周期性变化。"""

    def __init__(self, sr: int, B: int,
                 hz: float = 8.0, depth: float = 0.5):
        self.sr = int(sr); self.B = int(B)
        self.hz = float(hz); self.depth = float(np.clip(depth, 0.0, 1.0))
        self._phase = 0.0

    def reset(self):
        self._phase = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        inc = 2.0 * np.pi * self.hz / self.sr
        n = x.size
        ph = self._phase + inc * np.arange(n)
        self._phase = float((self._phase + inc * n) % (2.0 * np.pi))
        mod = 1.0 - self.depth * (0.5 - 0.5 * np.cos(ph).astype(np.float32))
        return (x * mod).astype(np.float32)


# =============================================================================
# 14. Vibrato —— 震音（延迟被 LFO 调制）
# =============================================================================
class Vibrato:
    """环形缓冲 + LFO 调制的延迟读位置。"""

    def __init__(self, sr: int, B: int,
                 hz: float = 7.0, depth_ms: float = 3.0, mix: float = 1.0):
        self.sr = int(sr); self.B = int(B)
        self.hz = float(hz)
        self.depth_samp = max(1, int(sr * depth_ms / 1000.0))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self._buf = np.zeros(self.depth_samp * 4 + 32, dtype=np.float32)
        self._p = 0
        self._phase = 0.0

    def reset(self):
        self._buf.fill(0); self._p = 0; self._phase = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        L = self._buf.size
        inc = 2.0 * np.pi * self.hz / self.sr
        n = x.size
        depth = self.depth_samp
        # 读位置（向量化）
        t = np.arange(n)  # int
        ph = self._phase + inc * t
        lfo = depth * (0.5 - 0.5 * np.cos(ph))
        # 写入（先写，因为读位置可能指到当前正在写入的位置）
        wp_idx = (self._p + t) % L
        self._buf[wp_idx] = x
        # 读位置（环形）
        r_f = (self._p + t) - (depth + lfo)
        r_f_mod = np.mod(r_f, L)
        r0 = np.floor(r_f_mod).astype(np.int64)
        r1 = (r0 + 1) % L
        frac = (r_f_mod - r0).astype(np.float32)
        v = self._buf[r0] * (1.0 - frac) + self._buf[r1] * frac
        self._p = int((self._p + n) % L)
        self._phase = float((self._phase + inc * n) % (2.0 * np.pi))
        m = self.mix
        return ((1.0 - m) * x + m * v).astype(np.float32)


# =============================================================================
# 15. BreathNoise —— 呼吸声
# =============================================================================
class BreathNoise:
    """包络保留 + 高频噪声 → 像在说话时能听到气流声。

    strength 0~1：混合的高频噪声量。
    """

    def __init__(self, sr: int, B: int, hp_cut: float = 2000.0):
        self.sr = int(sr); self.B = int(B)
        self.hp_cut = float(hp_cut)
        self._rng = np.random.default_rng()

    def reset(self):
        pass

    def process(self, x: np.ndarray, strength: float = 0.3) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        s = float(np.clip(strength, 0.0, 1.0))
        n = x.size
        env = np.abs(x)
        noise = self._rng.standard_normal(n).astype(np.float32) * env * 0.5
        hp = _iir_hp(noise, self.hp_cut, self.sr)
        return ((1.0 - s * 0.5) * x + s * hp).astype(np.float32)


# =============================================================================
# SilenceSaver —— 静音时跳过 DSP，节省算力
# =============================================================================
class SilenceSaver:
    """RMS 阈值判定：低于阈值连续 hold_ms → 输出静音；
    高于阈值 → 恢复（带 fade-in 防止咔哒声）。

    实际省 CPU 效果：在用户说话间隙可以让上层 effect 链完全跳过。
    这里是「软实现」：返回静音信号，让声音听感上是真的「关」了，但 DSP
    代码还是跑了（要彻底跳过要上层配合判断）。要做到真正的省 CPU，
    可以再加一个 `is_silent` 状态查询。
    """

    def __init__(self, sr: int, B: int,
                 threshold_db: float = -55.0,
                 hold_ms: float = 200.0, fade_ms: float = 30.0):
        self.sr = int(sr); self.B = int(B)
        self.threshold = 10 ** (float(threshold_db) / 20.0)
        self.hold_samps = int(sr * hold_ms / 1000.0)
        self.fade_samps = max(1, int(sr * fade_ms / 1000.0))
        self._silent_count = self.hold_samps  # 初始静音 → 第一帧就关
        self._fading_in = False
        self._fade_pos = 0

    def reset(self):
        self._silent_count = self.hold_samps
        self._fading_in = False
        self._fade_pos = 0

    @property
    def is_silent(self) -> bool:
        return self._silent_count < self.hold_samps and not self._fading_in

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < self.threshold:
            self._silent_count += x.size
        else:
            self._silent_count = 0
            self._fading_in = True
            self._fade_pos = 0
        if self._silent_count >= self.hold_samps and not self._fading_in:
            return np.zeros_like(x)
        # fade in
        n = x.size
        out = x.copy()
        if self._fading_in:
            f_len = min(self.fade_samps - self._fade_pos, n)
            if f_len > 0:
                ramp = np.linspace(self._fade_pos / self.fade_samps,
                                   (self._fade_pos + f_len) / self.fade_samps,
                                   f_len, dtype=np.float32)
                out[:f_len] *= np.clip(ramp, 0.0, 1.0)
                self._fade_pos += f_len
                if self._fade_pos >= self.fade_samps:
                    self._fading_in = False
        return out.astype(np.float32)


# =============================================================================
# EffectRack —— 总线
# =============================================================================
class EffectRack:
    """12+ 效果器 rack。config 格式：{effect_name: {enabled, ...params}}"""

    def __init__(self, sr: int, B: int):
        self.sr = int(sr); self.B = int(B)
        self.noise_gate = NoiseGate(sr, B)
        self.robot = Robot(sr, B)
        self.whisper = Whisper(sr, B)
        self.telephone = Telephone(sr, B)
        self.distortion = Distortion(sr, B)
        self.bitcrush = Bitcrush(sr, B)
        self.chorus = Chorus(sr, B)
        self.echo = Echo(sr, B)
        self.reverb = Reverb(sr, B)
        self.tone_eq = ToneEQ(sr, B)
        self.compressor = Compressor(sr, B)
        self.output_gain = OutputGain(sr, B)
        self.tremolo = Tremolo(sr, B)
        self.vibrato = Vibrato(sr, B)
        self.breath = BreathNoise(sr, B)
        self.silence_saver = SilenceSaver(sr, B)

    def reset(self):
        for name in EFFECT_NAMES + ("silence_saver",):
            getattr(self, name).reset()

    def process(self, x: np.ndarray, config: dict | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        config = config or {}

        # 顺序：noise_gate → 主体效果（robot/whisper/...）→ breath → 输出调整
        if config.get("noise_gate", {}).get("enabled"):
            x = self.noise_gate.process(x)
        if config.get("robot", {}).get("enabled"):
            c = config["robot"]
            self.robot.hz = float(c.get("hz", self.robot.hz))
            x = self.robot.process(x)
        if config.get("whisper", {}).get("enabled"):
            c = config["whisper"]
            x = self.whisper.process(x, strength=c.get("strength", 1.0))
        if config.get("telephone", {}).get("enabled"):
            x = self.telephone.process(x)
        if config.get("distortion", {}).get("enabled"):
            c = config["distortion"]
            x = self.distortion.process(x, drive=c.get("drive", 2.0), mix=c.get("mix", 1.0))
        if config.get("bitcrush", {}).get("enabled"):
            c = config["bitcrush"]
            x = self.bitcrush.process(x, strength=c.get("strength", 1.0))
        if config.get("chorus", {}).get("enabled"):
            c = config["chorus"]
            x = self.chorus.process(x, mix=c.get("mix", 0.4))
        if config.get("echo", {}).get("enabled"):
            c = config["echo"]
            self.echo.delay_ms = float(c.get("delay_ms", self.echo.delay_ms))
            self.echo.feedback = float(c.get("feedback", self.echo.feedback))
            self.echo.mix = float(c.get("mix", self.echo.mix))
            x = self.echo.process(x)
        if config.get("reverb", {}).get("enabled"):
            c = config["reverb"]
            x = self.reverb.process(x, mix=c.get("mix", 0.3), room=c.get("room", 1.0))
        if config.get("tone_eq", {}).get("enabled"):
            c = config["tone_eq"]
            x = self.tone_eq.process(x,
                                     bass_db=c.get("bass_db", 0.0),
                                     mid_db=c.get("mid_db", 0.0),
                                     treble_db=c.get("treble_db", 0.0))
        if config.get("compressor", {}).get("enabled"):
            x = self.compressor.process(x)
        if config.get("tremolo", {}).get("enabled"):
            c = config["tremolo"]
            self.tremolo.hz = float(c.get("hz", self.tremolo.hz))
            self.tremolo.depth = float(c.get("depth", self.tremolo.depth))
            x = self.tremolo.process(x)
        if config.get("vibrato", {}).get("enabled"):
            c = config["vibrato"]
            self.vibrato.hz = float(c.get("hz", self.vibrato.hz))
            x = self.vibrato.process(x)
        if config.get("breath", {}).get("enabled"):
            c = config["breath"]
            x = self.breath.process(x, strength=c.get("strength", 0.3))
        if config.get("output_gain", {}).get("enabled"):
            c = config["output_gain"]
            x = self.output_gain.process(x, gain_db=c.get("gain_db", 0.0))

        # SilenceSaver：放在最后（对整个效果链后的输出做静音检测）
        if config.get("silence_saver", {}).get("enabled"):
            x = self.silence_saver.process(x)
        return x


def build_effect_rack(sr: int, B: int) -> EffectRack:
    return EffectRack(sr, B)
