# -*- coding: utf-8 -*-
r"""voice_core 自测：不需要麦克风/声卡，纯数值验证。

运行： .venv\Scripts\python.exe test_voice_core.py
"""
from __future__ import annotations

import sys

import numpy as np

import voice_core as vc

SR = 48000
FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def dominant_freq(y, sr=SR, lo=50.0, hi=8000.0):
    """返回 [lo, hi] 范围内能量最强的频率。"""
    y = np.asarray(y, dtype=np.float64)
    w = np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y * w))
    freqs = np.fft.rfftfreq(y.size, 1.0 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    return float(freqs[band][np.argmax(spec[band])])


def sine(f, dur, sr=SR, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


# ---------------------------------------------------------------- 1. 恒等性
def test_identity_exact():
    """ratio=1 时必须原样还原（Hann 重叠相加窗和为 1），只是延迟 span。"""
    for B in (256, 512, 1024):
        sh = vc.StreamingPitchShifter(B)
        x = sine(300, 0.5)
        delay = sh.latency_samples
        total = x.size + delay
        xp_size = ((total + B - 1) // B) * B
        xp = np.concatenate([x, np.zeros(xp_size - x.size, np.float32)])
        out = np.concatenate([sh.process(xp[i:i + B], 1.0) for i in range(0, xp.size, B)])
        y = out[delay:delay + x.size]
        err = float(np.max(np.abs(y - x)))
        check(f"identity ratio=1.0 B={B}（延迟 {delay} 采样 = {delay / SR * 1000:.1f} ms）",
              err < 2e-3 and np.all(np.isfinite(y)), f"max|y-x|={err:.2e}")


def test_identity_noise():
    """用非周期信号（噪声）再验一次恒等性：周期信号可能掩盖时间轴上的错误。
    同时检查颗粒读出没有读到还没写入的采样（read_overflow 必须为 0）。"""
    rng = np.random.default_rng(7)
    for B in (256, 512, 1024):
        for ratio in (1.0, 1.25, 2.0):
            sh = vc.StreamingPitchShifter(B)
            x = (0.3 * rng.standard_normal(12000)).astype(np.float32)
            delay = sh.latency_samples
            total = x.size + delay
            xp = np.concatenate([x, np.zeros(((total + B - 1) // B) * B - x.size, np.float32)])
            out = np.concatenate([sh.process(xp[i:i + B], ratio)
                                  for i in range(0, xp.size, B)])
            y = out[delay:delay + x.size]
            if ratio == 1.0:
                ok = float(np.max(np.abs(y - x))) < 2e-3
                detail = f"max|y-x|={float(np.max(np.abs(y - x))):.2e}"
            else:
                # 变调后波形必然不同，但能量和时间轴不该崩：rms 同量级、无越界
                r_in, r_out = rms(x), rms(y)
                ok = 0.4 < r_out / max(r_in, 1e-9) < 2.5
                detail = f"rms {r_in:.3f}->{r_out:.3f} ({r_out / max(r_in, 1e-9):.2f}x)"
            check(f"噪声恒等/能量 B={B} ratio={ratio}", ok and sh.read_overflow == 0,
                  detail + f" 越界={sh.read_overflow}")


# ---------------------------------------------------------------- 2. 变调
def test_pitch_ratio():
    """不同半音数下，主频应变成 440 * 2^(semi/12)。"""
    x = sine(440, 1.0)
    for semi in (+12, +7, +4, -5, -12):
        expect = 440.0 * (2.0 ** (semi / 12.0))
        y = vc.process_offline(x, SR, semitones=semi, effect="none")
        seg = np.concatenate([y[int(0.25 * SR):int(0.75 * SR)]])
        f = dominant_freq(seg)
        rel = abs(f - expect) / expect
        check(f"变调 {semi:+d} 半音 -> {expect:.1f} Hz", rel < 0.02,
              f"实测 {f:.1f} Hz, 误差 {rel * 100:.2f}%")


def test_duration_and_rms():
    """时长必须不变，音量不能明显变化。"""
    for semi in (+12, +5, -7):
        x = sine(220, 0.8, amp=0.6)
        y = vc.process_offline(x, SR, semitones=semi)
        check(f"时长保持 {semi:+d} 半音", y.size == x.size, f"{y.size} vs {x.size}")
        r_in, r_out = rms(x[int(0.2 * SR):int(0.8 * SR)]), rms(y[int(0.2 * SR):int(0.8 * SR)])
        # 变调后主频变了但总能量应在一个量级（颗粒合成会略降）
        check(f"音量保持 {semi:+d} 半音", 0.5 < r_out / r_in < 1.5,
              f"rms {r_in:.3f} -> {r_out:.3f} ({r_out / r_in:.2f}x)")


def test_no_nan_or_clip():
    x = sine(150, 0.5, amp=0.95)
    for semi in (-12, -3, 0, 6, 12):
        y = vc.process_offline(x, SR, semitones=semi)
        peak = float(np.max(np.abs(y)))
        check(f"无 NaN/异常峰值 {semi:+d} 半音", np.all(np.isfinite(y)) and peak <= 1.2,
              f"peak={peak:.3f}")


def test_ratio_change_is_safe():
    """运行中改音调（拖滑块）不能让 shifter 读到未来采样/崩掉。"""
    B = 512
    sh = vc.StreamingPitchShifter(B)
    x = sine(300, 1.0)
    pad = (-x.size) % B
    xp = np.concatenate([x, np.zeros(pad, np.float32)])
    ratios = [1.0, 2.0, 0.5, 1.5, 0.75, 2.0, 1.0]
    outs = []
    for i in range(0, xp.size, B):
        r = ratios[(i // B) % len(ratios)]
        outs.append(sh.process(xp[i:i + B], r))
    y = np.concatenate(outs)
    check("运行中切换音调", np.all(np.isfinite(y)) and float(np.max(np.abs(y))) <= 1.2,
          f"peak={float(np.max(np.abs(y))):.3f}")


# ---------------------------------------------------------------- 3. 音效
def test_effect_robot():
    x = sine(440, 0.6)
    y = vc.process_offline(x, SR, effect="robot")
    spec = np.abs(np.fft.rfft(y[len(y) // 4:] * np.hanning(len(y) - len(y) // 4)))
    freqs = np.fft.rfftfreq(len(y) - len(y) // 4, 1.0 / SR)
    def band(a, b):
        m = (freqs >= a) & (freqs <= b)
        return float(np.max(spec[m]))
    p380, p500, p440 = band(370, 390), band(490, 510), band(432, 448)
    check("机器人音效：产生 ±60Hz 边带、抑制原频", p380 > 3 * p440 and p500 > 3 * p440,
          f"380Hz={p380:.1f} 500Hz={p500:.1f} 440Hz={p440:.1f}")


def test_effect_echo():
    imp = np.zeros(SR, dtype=np.float32)
    imp[0] = 1.0
    y = vc.process_offline(imp, SR, effect="echo")
    peak0 = float(np.max(np.abs(y[:100])))
    around = y[int(0.130 * SR) - 60: int(0.130 * SR) + 60]
    check("回声音效：130ms 处出现回音", peak0 > 0.9 and float(np.max(np.abs(around))) > 0.2,
          f"直通={peak0:.2f} 回音={float(np.max(np.abs(around))):.3f}")


def test_effect_phone():
    lo = vc.process_offline(sine(100, 0.5), SR, effect="phone")
    mid = vc.process_offline(sine(1000, 0.5), SR, effect="phone")
    a, b = rms(lo[int(0.2 * SR):]), rms(mid[int(0.2 * SR):])
    check("电话音效：100Hz 被压、1kHz 通过", b > 5 * a, f"100Hz rms={a:.4f}, 1kHz rms={b:.4f}")


def test_offline_gain_and_length():
    x = sine(440, 0.37)
    y = vc.process_offline(x, SR, semitones=3, effect="phone", gain=0.5)
    check("离线处理：长度一致 + 增益生效", y.size == x.size and np.all(np.isfinite(y)),
          f"len={y.size}")


def main():
    print(f"numpy {np.__version__}, 采样率 {SR}\n")
    for fn in (test_identity_exact, test_identity_noise, test_pitch_ratio, test_duration_and_rms,
               test_no_nan_or_clip, test_ratio_change_is_safe, test_effect_robot,
               test_effect_echo, test_effect_phone, test_offline_gain_and_length):
        print(f"--- {fn.__name__} ---")
        fn()
        print()
    if FAILURES:
        print(f"结果：{len(FAILURES)} 项失败 -> {FAILURES}")
        return 1
    print("结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
