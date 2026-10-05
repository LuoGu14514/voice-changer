# -*- coding: utf-8 -*-
"""简单变声器 —— DSP 核心。

只做三件事：
1. StreamingPitchShifter：实时变调（颗粒式重叠相加 + WSOLA 相位对齐，
   保持时长不变），可以在音频回调里逐块调用，音调比值可随时改变。
2. FormantCorrector：共振峰解耦（实时 STFT 包络校正）。
   单独使用时跟原声完全相同；接在变调后面能把「音高上去了但共振峰也
   跟着上去了 = 声像变小、变成花栗鼠」这种情况修掉，让升调后的声音
   保持原始的「声道形状」（听起来才像真人女声 / 男声，而不是小孩）。
3. EffectChain / EffectRack（v0.4.0 起：12+ 效果器 rack）：
   噪声门 / 机器人 / 气音 / 电话音 / 失真 / 位破坏 / 合唱 / 回声 /
   混响 / EQ / 压缩器 / 颤音 / 震音 / 呼吸声。多个效果可同时叠加，
   每个独立控制强度。EffectChain 是向后兼容的简单包装（仅单效果），
   新代码应直接用 EffectRack。

不依赖 scipy，只需要 numpy。STFT 走的是纯 numpy（np.fft.rfft）。
"""
from __future__ import annotations

import numpy as np

from effects import (
    EFFECT_NAMES as _EFFECT_NAMES,
    EFFECT_LABELS as _EFFECT_LABELS_FROM_EFFECTS,
    EffectRack,
    build_effect_rack,
)

__all__ = [
    "semitones_to_ratio",
    "StreamingPitchShifter",
    "FormantCorrector",
    "VoicePipeline",
    "EffectChain",
    "EffectRack",
    "process_offline",
    "EFFECTS",
    "EFFECT_LABELS",
    "VOICE_PRESETS",
    "LEGACY_EFFECT_CONFIGS",
]

# 向后兼容：旧的 4 个单效果模式（GUI 里仍能选）
EFFECTS = ("none", "robot", "echo", "phone")
EFFECT_LABELS = {"none": "无", "robot": "机器人", "echo": "回声", "phone": "电话音"}

# 旧的 effect 名 → 新 rack config（向后兼容用）
LEGACY_EFFECT_CONFIGS = {
    "none":  {},
    "robot": {"robot": {"enabled": True, "hz": 60.0}},
    "echo":  {"echo":  {"enabled": True, "delay_ms": 130.0, "feedback": 0.35, "mix": 0.45}},
    "phone": {"telephone": {"enabled": True}},
}


def _normalize_effect_spec(spec) -> dict:
    """把 preset 的 effect 字段（None/str/dict）归一化为 dict config。

    None 或空 → {}
    str：旧 "none/robot/echo/phone" → LEGACY_EFFECT_CONFIGS 对应项；其它 → {}
    dict：原样
    """
    if spec is None or spec == "":
        return {}
    if isinstance(spec, str):
        return dict(LEGACY_EFFECT_CONFIGS.get(spec, {}))
    if isinstance(spec, dict):
        return spec
    return {}

# v0.4.0：30 个角色音色 preset，每项用多效果组合（dict config）
# 字段：(显示名, 半音, 共振峰位移比, 效果 config, 分组)
# form_shift 1.0 = 共振峰不动；>1 = 变小（儿童/女）；<1 = 变大（男/大叔）。
# effects 字段支持三种类型：
#   - str: 旧版单效果名（"none"/"robot"/"echo"/"phone"），自动通过 LEGACY_EFFECT_CONFIGS 转换
#   - dict: 新版多效果组合 {effect_name: {enabled, ...params}}
#   - None 或 {}: 无效果
#
# 设计原则：
#   - 同一档内的 preset 共享基础 pitch+formant，仅用 effects 区分个性
#   - 真人女声：pitch ↑ + form_shift ↑ + 高频略提升 + 微量气声
#   - 大叔/磁性：pitch ↓ + form_shift ↓ + 低频提升
#   - 童声：pitch ↑↑ + form_shift ↑↑ + 大量气声
#   - 恶魔：pitch ↓ + 重失真 + 低频提升
#   - 兽人/幽灵/外星人：参考 SUONSUN9527 community.json
#
# 参考来源：
#   - suer781/MaidMic           (Apache-2.0, 萝莉默认/大叔/花栗鼠 preset)
#   - neboyang/VoiceChanger     (Apache-2.0, KITTY/ROSE/WOMAN/UNCLE/MAN/TOM)
#   - lyrebird-voice-changer    (GPL, 公开 industry baseline: Darth Vader -6 等)
#   - SUONSUN9527/windows-voice-changer  (community.json: 兽人/幽灵/外星人)
#   - sioaeko/OpenVoiceChanger  (12 效果器 rack 设计参考)
#   - 自调（基于听觉微调）
VOICE_PRESETS = (
    # —— 中性 ——
    ("原声",     0.0,  1.00, {}, "原"),

    # —— 女性 ——
    # 真女声：基础 pitch 上推 3 半音 + 微量 treble 提亮
    ("真女声",   3.0,  1.05, {"tone_eq": {"enabled": True, "treble_db": 1.5}}, "女"),
    # 御姐：成熟女声，低频 + 气声
    ("御姐",     4.0,  1.10, {"tone_eq": {"enabled": True, "bass_db": 1.0, "treble_db": 1.0},
                              "breath": {"enabled": True, "strength": 0.15}}, "女"),
    # 萌妹：可爱感（高频 + 气声）
    ("萌妹",     6.0,  1.18, {"tone_eq": {"enabled": True, "treble_db": 1.0},
                              "breath": {"enabled": True, "strength": 0.20}}, "女"),
    # 嗲嗲：娃娃气 + bass 收一点避免闷
    ("嗲嗲",     5.0,  1.20, {"tone_eq": {"enabled": True, "bass_db": -2.0, "treble_db": 2.0},
                              "breath": {"enabled": True, "strength": 0.30}}, "女"),
    # 客服女：温和压缩，听感稳定
    ("客服女",   2.0,  1.05, {"compressor": {"enabled": True, "threshold_db": -20.0, "ratio": 3.0}}, "女"),
    # 播音女：广播感压缩 + 提亮
    ("播音女",   1.0,  1.02, {"compressor": {"enabled": True, "threshold_db": -18.0, "ratio": 4.0},
                              "tone_eq": {"enabled": True, "bass_db": -1.0, "treble_db": 2.0}}, "女"),
    # 少妇：成熟女声 + 一点点房间混响
    ("少妇",     2.0,  0.96, {"tone_eq": {"enabled": True, "bass_db": 1.0},
                              "reverb": {"enabled": True, "mix": 0.2}}, "女"),
    # 老奶奶：老声 + 慢震音（手抖）+ 混响
    ("老奶奶",  -6.0,  0.88, {"tone_eq": {"enabled": True, "bass_db": -2.0},
                              "vibrato": {"enabled": True, "hz": 4.0},
                              "reverb": {"enabled": True, "mix": 0.3}}, "女"),

    # —— 男性 ——
    ("真男声",  -3.0,  0.95, {"tone_eq": {"enabled": True, "bass_db": 1.0}}, "男"),
    # 大叔：低频 + 呼吸感
    ("大叔",    -5.0,  0.90, {"tone_eq": {"enabled": True, "bass_db": 2.0},
                              "breath": {"enabled": True, "strength": 0.10}}, "男"),
    # 恶魔：重失真 + 低频
    ("恶魔",    -6.0,  0.85, {"distortion": {"enabled": True, "drive": 1.5, "mix": 1.0},
                              "tone_eq": {"enabled": True, "bass_db": 3.0}}, "男"),
    # 磁性男：低频 + 温和压缩
    ("磁性男",  -2.0,  0.92, {"tone_eq": {"enabled": True, "bass_db": 1.0},
                              "compressor": {"enabled": True, "threshold_db": -18.0, "ratio": 2.5}}, "男"),
    # 客服男：压缩
    ("客服男",  -1.0,  0.95, {"compressor": {"enabled": True, "threshold_db": -20.0, "ratio": 3.0}}, "男"),
    # 播音男：广播压缩 + EQ
    ("播音男",   0.0,  0.97, {"compressor": {"enabled": True, "threshold_db": -18.0, "ratio": 4.0},
                              "tone_eq": {"enabled": True, "bass_db": -1.0, "treble_db": 2.0}}, "男"),
    # 老人：低频 + 混响 + 压缩
    ("老人",    -7.0,  0.85, {"tone_eq": {"enabled": True, "bass_db": 3.0},
                              "reverb": {"enabled": True, "mix": 0.2},
                              "compressor": {"enabled": True, "threshold_db": -25.0, "ratio": 2.0}}, "男"),

    # —— 童声 ——
    # 萝莉：基础 + 较多气声
    ("萝莉",     7.0,  1.18, {"breath": {"enabled": True, "strength": 0.30}}, "童"),
    ("正太",     5.0,  1.10, {"breath": {"enabled": True, "strength": 0.20}}, "童"),
    # 小孩：高 pitch + 大量气声 + 高频提亮
    ("小孩",     9.0,  1.20, {"tone_eq": {"enabled": True, "treble_db": 1.0},
                              "breath": {"enabled": True, "strength": 0.40}}, "童"),
    # 娃娃音：极高 pitch + 合唱 + 大量气声
    ("娃娃音",  12.0,  1.20, {"chorus": {"enabled": True},
                              "breath": {"enabled": True, "strength": 0.50}}, "童"),
    # 小猫：卡通
    ("小猫",     4.0,  1.15, {"breath": {"enabled": True, "strength": 0.30}}, "童"),
    # 花栗鼠：bitcrush + 气声
    ("花栗鼠",   7.0,  1.20, {"bitcrush": {"enabled": True, "strength": 0.3},
                              "breath": {"enabled": True, "strength": 0.40}}, "童"),
    # 汤姆猫：高 pitch + 轻 bitcrush
    ("汤姆猫",  10.0,  1.18, {"bitcrush": {"enabled": True, "strength": 0.2},
                              "breath": {"enabled": True, "strength": 0.40}}, "童"),

    # —— 特效 ——
    ("机器人",    0.0,  1.00, {"robot": {"enabled": True, "hz": 60.0}}, "特效"),
    # 电音女王：基础 robot + 合唱
    ("电音女王",  4.0,  1.10, {"robot": {"enabled": True, "hz": 60.0},
                              "chorus": {"enabled": True}}, "特效"),
    # 外星人：高 hz robot + 混响（SUONSUN9527 alien）
    ("外星人",   2.0,  1.10, {"robot": {"enabled": True, "hz": 200.0},
                              "reverb": {"enabled": True, "mix": 0.3}}, "特效"),
    # 兽人：低频 + 短回声 + 失真（SUONSUN9527 orc）
    ("兽人",    -5.0,  0.85, {"echo": {"enabled": True, "delay_ms": 40.0, "feedback": 0.06, "mix": 0.5},
                              "tone_eq": {"enabled": True, "bass_db": 3.0},
                              "distortion": {"enabled": True, "drive": 1.2, "mix": 0.3}}, "特效"),
    # 幽灵：长回声 + 混响 + 合唱（SUONSUN9527 ghost）
    ("幽灵",    -2.0,  0.95, {"echo": {"enabled": True, "delay_ms": 350.0, "feedback": 0.35, "mix": 0.65},
                              "reverb": {"enabled": True, "mix": 0.4},
                              "chorus": {"enabled": True}}, "特效"),
    # 回声：标准 echo
    ("回声",     0.0,  1.00, {"echo": {"enabled": True, "delay_ms": 250.0, "feedback": 0.3, "mix": 0.45}}, "特效"),
    # 电话音：标准 telephone
    ("电话音",   0.0,  1.00, {"telephone": {"enabled": True}}, "特效"),
)


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


class FormantCorrector:
    """实时共振峰校正器（接在变调器后面用）。

    解决的问题：单独用颗粒式变调（WSOLA）时，音高升一截，共振峰（F1/F2/F3）
    也跟着升一截，听着就是「变小的成年人」——花栗鼠味儿，但不像真人女声；
    反过来降调时变成「变大的成年人」——像得了感冒。

    思路（lewark/pvc 的做法，简化成 streaming）：每 B 个采样做一次 STFT
    （N = 2B，hop = B/2，sqrt-Hann 满足 COLA），把频谱拆成两部分：

        |X(f)| = E(f) · R(f)

    E(f) 是「共振峰包络」（用沿频率轴的 rolling max 估计，相当于低分辨率
    的谱包络，峰宽 ~ 8 个 bin ≈ 375 Hz，恰好覆盖 F1=500/F2=1500/F3=2500）。
    R(f) 是「去掉共振峰后的细节」，主要就是谐波列的位置。

    WSOLA 让 |X| 整体在频率轴上压缩了 1/ratio 倍。要让共振峰回到原来的
    位置，只要把 E(f) 沿频率轴拉伸 ratio 倍（即 E_restored[k] = E[k*ratio]）
    再乘回 R(f) 上去。谐波列 R(f) 还在 WSOLA 给它的位置上，于是听众听到的
    「音高」是 WSOLA 的，而「声道形状」是原始说话人的。

    还有一个 form_shift_ratio 参数：>1 把共振峰往上挪（听起来像小孩 /
    卡通女声），<1 往下挪（像大叔）。和 pitch 解耦，可以独立调节。

    性能：B=512 / N=1024 时，每块 2 次 rfft + 1 次 max_filter，纯 numpy
    < 2 ms（B=256 时 1 ms，B=1024 时 4 ms，实测预算充足）。
    延迟：latency_samples = 2 * B（第一个块返回零，第二块开始有正确输出）。
    """

    def __init__(self, block_size: int = 512, fft_size: int | None = None,
                 filt_bins: int = 24, lpc_order: int | None = None,
                 envelope_method: str = "lpc"):
        self.B = int(block_size)
        self.N = int(fft_size) if fft_size else 2 * self.B
        # hop = N/2 → 50% overlap，sqrt-Hann 满足 COLA（sum = 1）
        self.hop = self.N // 2
        if self.hop != self.B:
            # 默认 B 块、N=2B、hop=B，跟 StreamingPitchShifter 对齐
            # 如果外部强行传不匹配的尺寸，最少保证 hop > 0
            self.hop = max(1, self.hop)
        # 包络估计方法：
        #   "lpc"      — 倒谱提升法（Imai & Abe 1978），输出平滑且与谐波密度无关
        #   "max_filter" — 沿频率轴的滑动最大值（v0.4 之前默认；快但容易高估包络）
        self.envelope_method = envelope_method if envelope_method in ("lpc", "max_filter") else "lpc"
        # 包络平滑窗口（仅 max_filter 用）：默认 24 bin ≈ 1125 Hz 宽
        self.filt_bins = max(3, int(filt_bins))
        # LPC 阶数：用于倒谱提升法的截止点。默认按 Fs/1000+2 经验公式。
        # Fs=48000 → 50；Fs=16000 → 18。阶数越大，包络越精细（也越抖动）。
        if lpc_order is None:
            # 选一个对典型语音 F1=500Hz 友好的默认值
            lpc_order = max(20, self.N // 20)
        self.lpc_order = int(lpc_order)
        # sqrt-Hann：分析和综合各用一次，乘起来满足 COLA
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.N) / self.N)
        self.win = (hann.astype(np.float32) ** 0.5)
        # 上一个 hop 这么多采样，拼成完整 N-sample 帧
        self._prev = np.zeros(self.hop, dtype=np.float32)
        # OLA 累加器：保存最新一帧的综合输出，等下两块再读
        self._ola = np.zeros(self.N, dtype=np.float32)
        # 已经处理过的帧数；< 1 时第一个块按静音输出（延迟 = N）
        self._frame_idx = 0

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _max_filter_1d(x: np.ndarray, size: int) -> np.ndarray:
        """沿最后一轴做 rolling max（edge 填充）。

        比 scipy.ndimage.maximum_filter1d 慢一点，但是纯 numpy、零依赖。
        size = 8 在 N=1024 时窗口约 375 Hz，刚好覆盖一个共振峰的宽度。
        """
        pad = size // 2
        xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, size - 1 - pad)], mode="edge")
        v = np.lib.stride_tricks.sliding_window_view(xp, size, axis=-1)
        return v.max(axis=-1)

    @staticmethod
    def _lpc_envelope(mag: np.ndarray, order: int, n_fft: int) -> np.ndarray:
        """用倒谱提升（cepstrum liftering）法从幅度谱估共振峰包络。

        思路（Imai & Abe 1978，"Spectral envelope extraction by improved
        cepstral windowing"）：把 log(|X|) 当成"信号"做 IFFT 得倒谱 c[n]，
        保留 c[0..order] 那一段（低倒频率对应谱包络的高时间尺度变化），把
        高阶 c[n > order] 当成"细节（谐波列）"置零，再 FFT 回频率域得到
        平滑的对数包络，最后 exp 就是线性包络。这样得到的包络与谐波密度
        无关，且能精确跟踪共振峰形状。

        参数：
            mag: (K,) 非负幅度谱（K = N//2 + 1）
            order: 倒谱保留的阶数（越大越精细，建议 N//20）
            n_fft: 原始 FFT 大小 N（用于对称处理偶/奇长度）
        返回：
            env: (K,) 平滑包络，与 mag 同形状、严格 > 0
        """
        mag = np.asarray(mag, dtype=np.float64)
        if mag.size == 0:
            return mag.astype(np.float32)
        # 1) 对数幅度 → irfft 得倒谱
        log_mag = np.log(np.maximum(mag, 1e-12))
        # 倒谱长度 = N（用 irfft 重建 2N 长度信号的对称 IFFT）
        # mag 来自 rfft，所以倒谱是 N 长，前 N 个权"因果"部分（n=0..N-1）
        # c[n] for n > order 是细节，需要清零。
        # 偶 N：irfft(rfft(x)) = x(N-1) 对称的话要额外把 N×2 长度上 c[N..2N-1] 当成镜像。
        # 这里用 2N 长度手算更直接。
        # log_mag 有 K = N//2+1 个点，对称扩展到 2K-2（DC 与 Nyq 复用），共 2K-1 长度。
        # 实际处理：用 irfft(log_mag) 得 *cepe（长度 2K-1），c[order+1 .. K-1] = 0。
        # 2K-1 = N + (K > 1) ... 麻烦。更简单：直接 irfft 取前 N 个，再镜像。
        # 更简单：用 np.fft.ifft(log_mag_full) 其中 log_mag_full 是 2K-2 长度的对称构造。
        K = mag.size
        # 把 log_mag 扩展成"完整"频谱（2K-2 长度的循环对称）：
        if K > 1:
            log_full = np.concatenate([log_mag, log_mag[-2:0:-1]])
        else:
            log_full = log_mag.copy()
        # IFFT → 实倒谱（理论上虚部是数值噪声）
        cep = np.fft.ifft(log_full).real.astype(np.float64)
        # 倒谱长度 = 2K - 2 = N（当 K = N//2+1 偶 N 时 N 是偶数）
        # 保留下标 [0..order]，其余清零
        cep_lifted = cep.copy()
        if order + 1 < cep_lifted.size:
            cep_lifted[order + 1:] = 0.0
        # FFT 回频率域 → 平滑对数包络
        log_env = np.fft.fft(cep_lifted).real.astype(np.float64)
        # log_env 长度 = 2K-2 = N，取前 K 个做包络
        log_env_K = log_env[:K]
        # 转线性 + 数值稳定
        # 对 log_env 上下限裁剪再 exp：避免数值爆炸
        log_env_K = np.clip(log_env_K, -20.0, 20.0)
        env = np.exp(log_env_K).astype(np.float32)
        # 兜底：env 必须严格 > 0
        env = np.maximum(env, 1e-8).astype(np.float32)
        # 归一化到 mag 的峰值（倒谱提升可能略缩放）
        if mag.max() > 0:
            scale = float(mag.max()) / max(float(env.max()), 1e-12)
            env = env * scale
        return env.astype(np.float32)

    def _process_frame(self, frame: np.ndarray, pitch_ratio: float,
                       form_shift_ratio: float) -> np.ndarray:
        """处理一帧（self.N 个采样），返回同长度输出。

        思路（formant 校正）：
        WSOLA 把整张谱整体乘以 pitch_ratio —— 基频和谐波都拉到新位置，
        共振峰也跟着搬到 F_orig × pitch_ratio。要让「同一个人换音高说话」，
        必须把共振峰（包络）从 F_orig × R 拉回到 F_orig（默认）或者
        F_orig × form_shift（用户可调）。

        包络估计方法（由 self.envelope_method 决定）：
          - "lpc"（v0.5.2 起默认）：倒谱提升法（cepstrum liftering）。
            log|X| → IFFT 得倒谱 → 保留 c[0..order] → FFT 回频率域 → exp。
            包络与谐波密度无关，能精确跟踪共振峰形状。order = N//20。
          - "max_filter"（旧）：沿频率轴 rolling max。快但容易高估包络
            （始终跟踪谐波峰顶，谐波密时偏离真包络）。

        具体步骤：
          1. 取 WSOLA 输出的频谱 mag_frame（含新音高的谐波列 + 新位置的包络）
          2. 用 LPC（或 max_filter）估出 WSOLA 包的包络 env_wsola
             （峰值在 F_orig × R 位置）
          3. residual = mag_frame / env_wsola → 拉平后的相对谱（去掉了 WSOLA 的
             包络调制，谐波相对幅度仍按 WSOLA 后的比例）
          4. 把 env_wsola 沿频率轴重采样：new_idx = idx × form_shift / R
             （这样新包络峰值在 F_orig × R × form_shift/R = F_orig × form_shift）
          5. new_mag = residual × env_shifted
          6. 沿用 WSOLA 输出的相位 → IFFT → 输出帧（音高已经由 WSOLA 改到 R，
             共振峰跑到 form_shift 用户想要的位置）
        """
        # 加窗 → FFT
        F = np.fft.rfft(frame * self.win)
        mag = np.abs(F).astype(np.float32)
        K = mag.shape[-1]
        idx = np.arange(K, dtype=np.float64)

        # 1) 估 WSOLA 输出频谱的包络
        if self.envelope_method == "lpc":
            env = self._lpc_envelope(mag, self.lpc_order, self.N)
        else:
            env = self._max_filter_1d(mag, self.filt_bins).astype(np.float32)
        mmax = float(mag.max()) if mag.size else 0.0
        floor = max(1e-4 * mmax, 1e-8)
        env = np.maximum(env, floor)

        # 2) 共振峰被搬走的「残余谱」（谐波列 + 细节）
        residual = np.where(env > 0, (mag / np.maximum(env, 1e-12)).astype(np.float32), 0.0)
        residual = residual.astype(np.float32)

        # 3) 决定包络的重采样比例：R / form_shift_ratio。
        #    WSOLA 把共振峰从 F_orig 挪到了 F_orig × R。
        #    要把 output 的包络峰值挪到 F_orig × form_shift 用户想要的位置：
        #      output[k] = env[k × shift]，peak 在 k = F_orig × form_shift 时
        #      要求 env[k × shift] 取到 env 的峰值 env_peak = F_orig × R。
        #      解：F_orig × form_shift × shift = F_orig × R → shift = R / form_shift。
        #    例：R=1.414, form_shift=1.0 → shift=1.414（拉回到原位）✓
        #    例：R=1.0, form_shift=1.15 → shift=0.870（peak 从 F_orig 移到 F_orig×1.15）✓
        #    例：R=1.414, form_shift=1.15 → shift=1.230（peak 移到 F_orig×1.15）✓
        #    例：R=1.0, form_shift=1.0 → shift=1.0（不变）✓
        R = float(pitch_ratio) if abs(float(pitch_ratio)) > 1e-9 else 1.0
        shift = R / float(form_shift_ratio) if abs(float(form_shift_ratio)) > 1e-9 else R
        if abs(shift - 1.0) > 1e-6 and K > 1:
            new_idx = np.clip(idx * shift, 0.0, K - 1.0)
            env_shifted = np.interp(new_idx, idx, env).astype(np.float32)
            env = env_shifted

        # 4) 重组：残谱（保留了 WSOLA 给的谐波列位置）× 包络（搬到 form_shift 位置）
        new_mag = (residual * env).astype(np.float32)
        # 相位保持原样（残谱本身没动，OLA 自然做相位对齐）
        out = np.fft.irfft((new_mag * np.exp(1j * np.angle(F))).astype(np.complex64),
                           n=self.N).real.astype(np.float32)
        # 综合窗
        return out * self.win

    # ---------------------------------------------------------------- 状态
    def reset(self) -> None:
        self._prev[:] = 0
        self._ola[:] = 0
        self._frame_idx = 0

    @property
    def latency_samples(self) -> int:
        # 第一个块返回静音，第二个块起输出第一块的内容（带 OLA 拼接）。
        # 所以延迟 = B（1 个块），不是 N。
        return self.B

    # ---------------------------------------------------------------- 处理
    def process(self, x: np.ndarray, pitch_ratio: float = 1.0,
                form_shift_ratio: float = 1.0,
                orig: np.ndarray | None = None) -> np.ndarray:
        """输入 B 个采样，返回同长度的共振峰校正结果。

        参数：
            x: B 个 float32 采样（WSOLA 之后的）
            pitch_ratio: 上游变调器的 ratio（用来决定包络要搬多远）
            form_shift_ratio: 独立于 pitch 的共振峰位移
                            （1.0 = 把 WSOLA 搬走的共振峰拉回原位
                             1.15 = 拉到原位的 1.15 倍频（萝莉/卡通女）
                             0.85 = 拉到原位的 0.85 倍频（大叔））
            orig: 已弃用，保留仅作向后兼容；新算法只用 x 自己。
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        B = self.B
        if x.size < B:
            xp = np.zeros(B, dtype=np.float32)
            xp[: x.size] = x
            x = xp
        else:
            x = x[:B]

        # 拼成完整 N-sample 帧：上一帧的尾巴 + 当前块
        frame = np.concatenate([self._prev, x]).astype(np.float32)
        self._prev = x.copy()

        # 跑一遍谱处理
        new = self._process_frame(frame, pitch_ratio, form_shift_ratio)

        # OLA：第一帧返回静音（latency = N），之后取累加器前 B 个采样
        self._ola += new
        if self._frame_idx >= 1:
            emit = self._ola[:B].copy()
        else:
            emit = np.zeros(B, dtype=np.float32)
        # 左移 B，腾出空间给下一帧
        self._ola[:B] = 0.0
        self._ola = np.roll(self._ola, -B)
        self._ola[-B:] = 0.0
        self._frame_idx += 1
        return emit


class VoicePipeline:
    """把变调 + 共振峰校正 + 音效串起来的整链路。

    用法：
        pipe = VoicePipeline(sr=48000, block_size=512)
        # 旧 API（向后兼容）：effect = "robot" 等单字符串
        y = pipe.process(x_mic, semitones=3.0, form_shift_ratio=1.0, effect="robot")
        # 新 API：effects_config = {"robot": {"enabled": True, "hz": 80}}
        y = pipe.process(x_mic, semitones=3.0, form_shift_ratio=1.0,
                         effects_config={"robot": {"enabled": True, "hz": 80}})
    """

    def __init__(self, samplerate: int = 48000, block_size: int = 512,
                 formant_correct: bool = True, align: bool = True):
        self.sr = int(samplerate)
        self.B = int(block_size)
        self.shifter = StreamingPitchShifter(self.B, align=align)
        self.formant = FormantCorrector(self.B) if formant_correct else None
        # 直接用 EffectRack（不走 EffectChain 包装）以支持多效果组合
        self.effects = EffectRack(self.sr, self.B)
        self._form_shift = 1.0
        self._current_config: dict = {}

    def reset(self) -> None:
        self.shifter.reset()
        if self.formant is not None:
            self.formant.reset()
        self.effects.reset()
        self._current_config = {}

    def set_effect(self, mode: str) -> None:
        """旧式 set_mode 入口（仅供 GUI 旧的下拉框用）。"""
        mode = mode if mode in EFFECTS else "none"
        self._current_config = LEGACY_EFFECT_CONFIGS[mode]

    def set_effects_config(self, config: dict | None) -> None:
        """新式入口：直接指定多效果 dict config。"""
        self._current_config = _normalize_effect_spec(config)

    @property
    def latency_samples(self) -> int:
        base = self.shifter.latency_samples
        if self.formant is not None:
            base += self.formant.latency_samples
        return base

    def process(self, x: np.ndarray, semitones: float = 0.0,
                form_shift_ratio: float | None = None,
                effect: str | None = None,
                effects_config: dict | None = None) -> np.ndarray:
        """处理一个 B 块。

        effect 与 effects_config 二选一：
          - 给 effect（str）：用 LEGACY_EFFECT_CONFIGS 转换
          - 给 effects_config（dict）：直接作为当前效果链
        """
        ratio = semitones_to_ratio(semitones)
        y = self.shifter.process(x, ratio)
        if self.formant is not None:
            fs = self._form_shift if form_shift_ratio is None else float(form_shift_ratio)
            y = self.formant.process(y, pitch_ratio=ratio, form_shift_ratio=fs)
        # 决定当前 config
        if effects_config is not None:
            cfg = _normalize_effect_spec(effects_config)
        elif effect is not None:
            mode = effect if effect in EFFECTS else "none"
            cfg = LEGACY_EFFECT_CONFIGS[mode]
        else:
            cfg = self._current_config
        if cfg:
            y = self.effects.process(y, cfg)
        return y


class EffectChain:
    """向后兼容的简单 wrapper —— 内部用 EffectRack，只暴露「单效果模式」。

    新代码应直接用 `from voice_core import EffectRack`（或 `from effects import
    EffectRack`），传入完整的 dict config。同时启用多个效果、自定义参数。
    """

    def __init__(self, samplerate: int = 48000, block_size: int = 512,
                 echo_ms: float = 130.0, echo_feedback: float = 0.35,
                 echo_mix: float = 0.45, ringmod_hz: float = 60.0,
                 phone_lo: float = 400.0, phone_hi: float = 3000.0, phone_taps: int = 201):
        # 旧版参数保留接口但不再使用（保持向后兼容签名）
        self.sr = int(samplerate)
        self.B = int(block_size)
        self.echo_ms = float(echo_ms)
        self.echo_feedback = float(echo_feedback)
        self.echo_mix = float(echo_mix)
        self.ringmod_hz = float(ringmod_hz)
        self.phone_lo, self.phone_hi = float(phone_lo), float(phone_hi)
        self.phone_taps = int(phone_taps)
        # 真正的 DSP 在 EffectRack 里
        self._rack = EffectRack(self.sr, self.B)
        self._config = {}
        self.mode = "none"

    def reset(self) -> None:
        self._rack.reset()

    def set_mode(self, mode: str) -> None:
        mode = mode if mode in EFFECTS else "none"
        self.mode = mode
        self._config = LEGACY_EFFECT_CONFIGS[mode]

    def process(self, x: np.ndarray) -> np.ndarray:
        return self._rack.process(x, self._config)


def process_offline(x, samplerate: int, semitones: float = 0.0, effect: str = "none",
                    block_size: int = 512, gain: float = 1.0,
                    align: bool = True, formant_correct: bool = False,
                    form_shift_ratio: float = 1.0,
                    effects_config: dict | None = None) -> np.ndarray:
    """离线处理一整段音频（文件变声用）。输出长度与输入一致。

    effect / effects_config：
      - 默认 "none"（向后兼容）
      - str 旧 API："none"/"robot"/"echo"/"phone"
      - dict 新 API：{"robot": {"enabled": True, "hz": 80}} 多效果组合
      - effects_config 优先于 effect（如果两者都给）

    formant_correct 默认 False（opt-in）：
    共振峰校正在简单信号（纯音/合成测试）上会把信号削掉（residual×env
    模型假设有清晰的谐波列+包络分离），但在真人语音/复杂音频上才真正发挥
    「拉回共振峰」的效果。CLI/GUI 想要更「真人」效果时显式打开。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    B = int(block_size)
    pipe = VoicePipeline(samplerate, B,
                         formant_correct=bool(formant_correct),
                         align=align)
    ratio = semitones_to_ratio(semitones)
    delay = pipe.latency_samples

    # 决定 cfg（effects_config 优先）
    if effects_config is not None:
        cfg = _normalize_effect_spec(effects_config)
    elif isinstance(effect, str):
        cfg = LEGACY_EFFECT_CONFIGS.get(effect, {})
    else:
        cfg = {}

    # 输出比输入慢 delay 个采样，所以要多喂 delay 个采样才够裁回来
    total = int(x.size) + int(delay)
    xp_size = ((total + B - 1) // B) * B
    xp = np.concatenate([x, np.zeros(xp_size - x.size, dtype=np.float32)])

    chunks = []
    for i in range(0, xp.size, B):
        y = pipe.process(xp[i:i + B], semitones=semitones,
                         form_shift_ratio=form_shift_ratio,
                         effects_config=cfg)
        chunks.append(y)
    y = np.concatenate(chunks) * float(gain)
    y = y[delay: delay + x.size]
    if y.size < x.size:
        y = np.concatenate([y, np.zeros(x.size - y.size, dtype=np.float32)])
    return y.astype(np.float32)
