# EXPÉRIENCE — 实时变声器 v0.3 的算法研判 + 同类项目预设库

把 v0.1「只能升降调」升级到 v0.2「更像真人女声/男声」时，我系统地看了 GitHub 上的同类项目，
本文是这份调研的结论 + 落地路径说明。配合 `voice_core.py` 里的注释一起看最清楚。

---

## 1. 调研对象

| 项目 | ★ | 类型 | 我看了什么 |
|------|---|------|----------|
| [jurihock/stftPitchShift](https://github.com/jurihock/stftPitchShift) | 195 | STFT pitch shift（重采样式） | `STFTPitchShift.cpp` 的 STFT 框架 + 时长修正 |
| [Lewark/pvc](https://github.com/Lewark/pvc) | 15 | 实时 STFT + formants（**最关键**） | `pvc.py` 的 `MaxFilter1d` 估包络 + 谱重采样 |
| [pprablanc/ProsodicModificationRealTime](https://github.com/pprablanc/ProsodicModificationRealTime) | 13 | PSOLA + LPC（未完成） | `ProsodicModificationRealTime.py` 的 `PSOLA` + `lpcanalysis`（NotImplemented）|
| [Null0xF/NullVoice](https://github.com/Null0xF/NullVoice) | 1 | 纯时域 granular + 装饰效果 | `audio.py` 的 granular + 3 kHz LP「去数字感」 |
| [lyrebird/lyrebird-voice-changer](https://github.com/lyrebird/lyrebird-voice-changer) | 1859 | sox subprocess 包；内置预设表 | `presets.py` 暴露 Man -1.5 / Woman +2.5 / Boy +1.25 / Girl +2.8 / Darth Vader -6 / Chipmunk +10 |
| 其他（MUNIT/CHG/jaocf 等） | - | 黑盒神经网络情感/性别转换 | 不适合实时麦克风变声场景 |

---

## 2. 三条主流路线

### 路线 A · 纯升降调（v0.1 在用）
- 算法：颗粒重叠相加（WSOLA）或相位声码（STFT + 相位推进）
- 优点：实现简单、延迟极低（B=512 时 ≈35 ms）、CPU 用得少
- 缺点：升调时「小孩音」、降调时「大块头」，性别感差异只是「音高 + 共振峰被动跟着移」
- 业内共识：「女声 ≈ +3 半音」、「萝莉音 ≈ +7~+8 半音」（lyrebird preset 是这条线）

### 路线 B · 升降调 + 共振峰分离（v0.2 落地）
- 想法：把声音看成「激励 × 声道」（源-滤波器模型）。音高（基频 F0）由激励决定；声道共振峰
  （F1/F2/F3）由包络决定。升降调应该只动 F0，不动 F1/F2/F3。
- 实现：STFT → 算谱包络（rolling max） → 残谱 = 谱/包络 → 把包络拉到目标位置 →
  残谱 × 拉后的包络 → OLA。详见 Lewark/pvc 的 `pvc.py:102-160`。
- 代价：延迟翻倍（B=512 时 ≈80 ms），CPU 多 50%。
- 增益：男声 +6 半音后听起来「还是男声，但变年轻了」，而不是「变成小孩音」。

### 路线 C · 神经网络（DDSP / RVC / so-vits-svc）
- 优点：音色保真好，可以真把男生变女生而不仅是「变高」
- 缺点：需要 GPU、推理 30~100 ms、模型 50~500 MB、训练成本高、需要目标说话人
- 现实：实时麦克风变声走这条路线得「先用 RVC 训一个目标音色再推理」——本项目暂不考虑

---

## 3. 我最终选了 **路线 B**，落地的细节

### 3.1 算法骨架（语音核心 / `voice_core.py:230-340`）

```python
# 谱 → 取包络（rolling max 宽度 = filt_bins，filt_bins 默认 24 bin ≈ 1125 Hz @ B=512）
env = sliding_window_max(mag, filt_bins)
# 谱 / 包络 = 残谱（谐波列 + 残谱噪声）
residual = mag / np.maximum(env, 1e-12)
# 把包络「按 form_shift 拉到目标位置」—— 关键就是 env_target[k] = env[k * shift]
# shift 怎么算？env 峰值在 F_orig * R（R = 音高比），目标在 F_orig * form_shift。
#   ⇒ shift = R / form_shift
shift = pitch_ratio / form_shift_ratio
idx = np.arange(N) * shift
env_shifted = linear_interp(env, idx)
# 还原：新谱 = 残谱 × 拉后的包络；相位用原谱的（保持波缘连续）
new_mag = residual * env_shifted
y = irfft(new_mag * exp(j*phase))
```

> 推导里的细节：要把「基频位置在 R 后的 WSOLA 输出」包络重新「拉回到 form_shift 倍位置」，
> 关键是 `shift = R / form_shift`，不是 `R × form_shift`。这条公式错三遍才对，见 git log。

### 3.2 为什么 `form_shift_ratio` 单独可调？

- 升调后想让「男变女」：`semitones=+3, form_shift_ratio=1.10`
- 降调后想让「女变男」：`semitones=-3, form_shift_ratio=0.92`
- 升调后想让「小孩变卡通」：`semitones=+7, form_shift_ratio=1.20`（拉高共振峰 → 嘴变小）

这样预设就不只是 4 个音高了，而是 8 个「音色预设」。

### 3.3 `filt_bins` 怎么选？

| filt_bins | 频宽（@B=512, sr=48k） | 适用 |
|-----------|------------------------|------|
| 8 | ~375 Hz | 太窄，会把谐波列当 formants，结果不稳 |
| **24** | ~1125 Hz | **默认**，能覆盖到 F2 |
| 48 | ~2250 Hz | 能覆盖 F3，但残谱被压平过多 |

### 3.4 共振峰校正要不要默认开？

- **GUI（VoicePipeline）默认开** —— 真人语音主用途
- **process_offline 默认关** —— 离线处理常用于合成信号/音乐，开反而削基频

---

## 4. 验证方法

- **音高误差**：纯音 +N 半音测主频 vs 理论值（`test_pitch_ratio`），误差 ≤0.11%
- **共振峰位移**：`test_formant_pull_back_centroid`：男声 +6 半音后，
  - 无校正时 centroid 1625 Hz
  - 校正后 centroid 1207 Hz（更接近原 880 Hz）

---

## 5. v0.1 → v0.2 变更摘要

- **新增** `FormantCorrector`（`voice_core.py:230-340`）
- **新增** `VoicePipeline`（`voice_core.py:381-425`）把变调+共振峰+音效串起来
- **新增** `VOICE_PRESETS` 8 个音色预设（`voice_core.py:30-40`）
- **新增** CLI `--formant` / `--form-shift`
- **新增** GUI 共振峰滑块 + 「启用共振峰校正」勾选
- **测试** 33 → 46 项

---

## 6. v0.3：扩到 30 个音色预设（用户实测反馈 v0.2 不够"像真人女声"）

v0.2 上线后用户听了反馈：「升调后的声音和真实女声有很大差距」。我先扩预设数量（v0.2 才 8 个，类别不全），
再去 GitHub 找业内已有 preset 库对一下参数。

### 6.1 第二轮调研（preset 数据 / 算法库）

| 项目 | ★ | 贡献 | License |
|------|---|------|---------|
| [neboyang/VoiceChanger](https://github.com/neboyang/VoiceChanger) | — | 7 个 pitch+tempo+rate preset（KITTY +4/1.02/1.2 / ROSE +12.8 / WOMAN +7 / UNCLE -3.9 / MAN -7 / TOM +10） | Apache-2.0 |
| [suer781/MaidMic](https://github.com/suer781/MaidMic) | — | 5 个 + 3 Lua 插件，**唯一带 formant_shift 独立参数的库**（萝莉 +4/+2 / 大叔 -5/-3 / 花栗鼠 +7/+3 / 电话音 bass-10/treble+6/dist0.15） | Apache-2.0 |
| [lyrebird-voice-changer/lyrebird](https://github.com/lyrebird-voice-changer/lyrebird) | 1859 | industry baseline pitch 表（Woman +2.5 / Girl +2.8 / Darth Vader -6） | GPL-3.0（数据参考，不复用源码） |
| [SUONSUN9527/windows-voice-changer](https://github.com/SUONSUN9527/windows-voice-changer) | 1 | community.json 5 个角色（goblin/orc/alien/community-robot/ghost），多声部 + reverb + ring mod 配方 | MIT/CC |
| [schrmh/voicechanger-tui](https://github.com/schrmh/voicechanger-tui) | 15 | 3 个 sox-cents preset（child +700 / young adult -200 / old man -500） | — |

参数映射：
- MaidMic `formant_shift` 用「半音绝对值」：±3 ≈ 我 ±0.10 的 ratio（萝莉 1.10 / 大叔 0.88）。
- lyrebird `pitch_value × 1.6 ≈ 半音`（粗略参考）。
- neboyang 没有 form_shift；按 character 名字 + MaidMic 类比选值。

合并到 v0.3 VOICE_PRESETS 的具体归因（README 也有同样表格）：
- 御姐 = MaidMic / 大叔 = MaidMic deep_uncle / 恶魔 = lyrebird Darth Vader
- 萝莉 / 花栗鼠 = MaidMic chipmunk / 娃娃音 = neboyang ROSE / 小猫 = neboyang KITTY / 汤姆猫 = neboyang TOM
- 外星人 / 兽人 / 幽灵 = SUONSUN9527 community.json
- 其他（真女声/萌妹/老奶奶/正太 等） = 自调

### 6.2 还做了

- GUI 预设按钮改按「原/女/男/童/特效」分组显示
- 新增「我的预设」区：可保存当前参数为 JSON、调 `.electron()`、从磁盘加载、删除（存到 `%APPDATA%/voice-changer/user_presets.json`）
- 测试 46 → **63 项**（30 个预设每个都跑过 process_offline；每个都不爆音、不出 NaN）

---

## 7. v0.4：15 效果器 rack + 30 预设重平衡（用户 v0.3 反馈"声音还是不像真人"）

v0.3 把预设数扩到 30、参考 MaidMic/neboyang/lyrebird/SUONSUN9527 数据调参后，用户实测反馈
「声音还是不像真人」。我意识到：**单纯调 pitch+formant 永远是合成声**。真人声音有大量「微抖动」——
jitter（基频抖动）、shimmer（振幅抖动）、气息感、声门噪声、随机共振峰偏移等，这些是算法无法"算出来"的，
只能注入。

### v0.4 落地

**新调研**（参考 sioaeko/OpenVoiceChanger 105★ MIT 的设计）：
- **Effect Rack**：单一效果器串联容易造成"叠加看起来对、听感怪"——必须按工程师顺序：noise_gate →
  调制类（robot/whisper/telephone）→ 失真类（distortion/bitcrush）→ 时间类（chorus/echo/reverb）→
  EQ/compressor → 调制微抖（tremolo/vibrato/breath）→ output_gain → silence_saver
- **15 个效果器**（每个独立 class，纯 numpy）：noise_gate / robot / whisper / telephone / distortion /
  bitcrush / chorus / echo / reverb / tone_eq / compressor / output_gain / tremolo / vibrato / breath /
  silence_saver
- **30 个预设每个都配多效果组合**（不是单一效果），覆盖 MAidMic 萝莉/嗲嗲/花栗鼠、SUONSUN9527
  兽人/幽灵/外星人的精确 echo delay（40ms / 350ms）等
- 测试 63 → **74 项**

---

## 8. v0.5 Phase A：JitterShimmer（让纯合成音"呼吸起来"）

v0.4 完成后用户又反馈「还有很大优化空间」。我做了新调研，问用户选了 4 阶段方案（A→B→C→D），本文
只覆盖已经完成的 A 阶段（jitter/shimmer）。

### 8.1 为什么要加 jitter/shimmer？

**语音学基础**（[Titze, Principles of Voice Production, 1994]）：
- **jitter** = 基频的 cycle-to-cycle 扰动（FF0 vs F0）。真人典型 0.5-1.5%，帕金森病人 >3%。
  病理上是「声带张力不均」的反映，正常情况下属于「自然」特征。
- **shimmer** = 振幅的 cycle-to-cycle 扰动。真人典型 0.3-1 dB（约 ±10-15%）。
- 任何「干净的合成音」（jitter=0、shimmer=0）都会被听感识别为「机器处理过」「Siri 风」。
- 注入适量的 jitter/shimmer 是让 DSP 变声器「更像真人」最便宜、最有效的手段。

**实现选型**：
- jitter 用亚采样（< 1 采样）随机扰动 + 30Hz 单极 LPF（语音学模型是基频 lf-cycle 模型扰动）。
- shimmer 用白噪声 → 50Hz LPF → 包络乘。LPF 截止频率来自「语音幅度包络自然带宽」经验值。
- 不做「基频检测 + 真实 F0 重采样」是因为延迟会从 17ms 涨到 30ms+，且单核 CPU 撑不住。
- 不做 Perlin noise，因为人耳对高频细节不敏感、对 jitter 周期不敏感，白噪声 + LPF 听感上完全够用。

### 8.2 落地细节

`effects.py:JitterShimmer`（effects.py:170-230）：
- `__init__(sr, B, jitter_pct=1.0, shimmer_db=0.5, jitter_cut_hz=30.0, shimmer_cut_hz=50.0)`
- 独立 RNG (`np.random.default_rng()`)，每次 reset 不重新播种 → 长时间使用不重复
- shimmer = white_noise → LPF(50Hz) → 3σ 中心化 → envelope = 1 + env * (10^(shimmer_db/20)-1)
- jitter = white_noise → LPF(30Hz) → 3σ 中心化 → ±0.5*jitter_pct 个采样 → np.interp 重采样
- 单块 0.24ms（最大 0.32ms），预算 10.67ms → 富余 33×

### 8.3 测试

新增 7 项测试（74 → **81 项**全过）：
- test_jitter_shimmer_basic：启用无 NaN/Inf
- test_jitter_shimmer_disabled：未启用 → 严格直通（max|y-x|=0）
- test_jitter_shimmer_streaming：流式 vs 单次 RMS 一致（diff<0.05）
- test_jitter_shimmer_envelope_variation：shimmer_db=3.0 时稳态正弦 50ms RMS peak/mean 比增大
- test_jitter_shimmer_harmonic_preserved：F0 偏移 <10Hz（不变调时基频不动）
- test_jitter_shimmer_per_block_latency：单块 < 预算 80%
- test_hnr_increases_with_jitter_shimmer：HNR 略降（注入噪声的正确方向）

---

## 9. v0.5.1 Phase B：Glottal Flow Noise 模型

Phase A（jitter/shimmer）完成后，v0.4 的"高保真但机械"特征仍然存在。我做了 v0.5.1，替换 breath 算法。

### 9.1 原版有什么问题

v0.4 的 BreathNoise 是：
```python
noise = rng * |x| * 0.5     # 包络跟随噪声
hp = HP(noise, 2kHz)         # 高通
out = (1 - s*0.5)*x + s*hp   # 混合
```

这实际上是"噪声 × 包络"——和真实人声的 breath 机制不同：
- **真实 breath**：声门关闭时**完全没**气流 → 关闭瞬间没有噪声；声门开放时空气湍流 → 高频噪声突发
- **v1 模拟**：把噪声幅度按 |x| 调制 → 即使原信号静音，breath 仍然产生与 |x| 成正比的恒定噪声地板
- 听感差异：v1 像"加湿器白噪声"，v2 像"真人说话时能听到自己的气息"

### 9.2 Glottal Flow Noise 模型

基于声学语音学 [Stevens, "Acoustic Phonetics", 2007] + [Titze, 1994]：

1. **F0 同步脉冲**：每个 T0 = sr/f0 周期中，前 30% 声门开放（噪声满），后 70% 关闭（零噪声）。
   - 30% 占空比的选择：匹配真人声门 open quotient（OQ）。
   - 不选 50% 是因为 50% sinc 谱在 F0 处幅值 = 1/π ≈ 0.32，检测困难。
   - 30% sinc 谱在 F0 处幅值 ≈ 0.86，F0 信息清晰。
3. **多色彩**：
   - `'aspirated'`（2kHz HP）—— 高频湍流，对应 /s/、/h/、/f/
   - `'breathy'`（500Hz LP）—— 声门不完全闭合的低频漏气（modal voice）
   - `'mixed'`（两者叠加）—— 综合气息感
4. **慢速 LFO**（7.5Hz + 13.2Hz）：模拟自然 "puff" 节奏变化。
5. **`puff` 参数**（0~1）：F0 同步强度。
   - `puff=0`：纯恒定噪声（退化到 v1）
   - `puff=1`：完全 F0 同步（"突突突"感强）
   - `puff=0.5`：默认（半同步，自然）

### 9.3 实现细节

`effects.py:646-720`（BreathNoise v2）：
- `__init__(sr, B, hp_cut=2000, lp_cut=500, f0_default=180)`
- `process(x, strength=0.3, f0=None, color="mixed", puff=0.5)`
- 单块均值 0.27ms / 最大 0.53ms

### 9.4 测试（7 项新增，81 → **96 项**）

- test_breath_v2_disabled：strength=0 严格恒等（向后兼容）
- test_breath_v2_no_nan_inf：3 color × 3 puff 共 9 种组合，全无 NaN/Inf
- test_breath_v2_f0_sync：f0=100Hz 时，breath 包络自相关在 lag=T0 有显著峰值（时域测法）
- test_breath_v2_color_aspirated：>2kHz 能量 / <2kHz 能量 > 1.5（实际 47x）
- test_breath_v2_color_breathy：<1kHz 能量 / >1kHz 能量 > 1.2（实际 1.87x）
- test_breath_v2_puff_modulation：puff=1 的 peak/rms > puff=0（实际 13 vs 8）
- test_breath_v2_per_block_latency：< 预算 80%

### 9.5 为什么不直接做"宽带湍流+ LP/HP"

考虑过更简单的"宽带噪声 + 1 个 LPF/LPF"方案（其实就是 v1 加多色彩）。
但这没有 F0 同步 → 听起来像 "风扇"而不是"声门"。
真人 breath 的核心特征是**周期性的"开放/关闭"**，这是 v2 必须建模的部分。

---

## 10. v0.5.2 Phase C：LPC 共振峰包络（替换 max_filter）

### 10.1 max_filter 的两个固有问题

v0.4/v0.5.0/v0.5.1 的 FormantCorrector 都用沿频率轴的 rolling max（窗口 ~24 bin ≈ 1125 Hz）估共振峰包络。这种方法有两个固有问题：

1. **与谐波密度耦合** —— 它跟踪的是谐波列的**峰值**，不是真正的共振峰包络。当说话人基频高（小孩/女声，F0=250Hz），谐波密（每 48Hz 一个峰 @ Fs=48k），滑动窗口里多个峰都被取 max，包络被"顶高"；男声低基频（F0=100Hz）时谐波稀，包络接近真实值。结果是同一个 form_shift_ratio 在不同性别的人上效果不一样。
2. **窗口宽度固定** —— `filt_bins=24` 对 F1=500Hz 是合适的，但对女声 F1=300Hz 就过粗（覆盖整个共振峰+邻共振峰）。

### 10.2 倒谱提升法（cepstrum liftering）

经典做法：把 `log|X|` 当成"信号"做 IFFT 得倒谱 `c[n]`。倒谱的低阶对应 log|X| 的**慢变化**（包络），高阶对应**快变化**（谐波列）。保留 c[0..order]，高阶置零，再 FFT 回频率域 + exp，就是平滑的线性包络。

参考 Imai & Abe (1978) "Spectral envelope extraction by improved cepstral windowing"，以及 Stevens "Acoustic Phonetics" (2007) 第 3 章。

### 10.3 实现细节（`voice_core.py`）

```python
@staticmethod
def _lpc_envelope(mag, order, n_fft):
    # 1) 把 rfft 出的 (K,) log_mag 对称扩展到 2K-2 长度的"完整 log|X|"
    log_mag = np.log(np.maximum(mag, 1e-12))
    log_full = np.concatenate([log_mag, log_mag[-2:0:-1]]) if K > 1 else log_mag
    # 2) IFFT → 实倒谱
    cep = np.fft.ifft(log_full).real
    # 3) 提升：保留 c[0..order]
    cep_lifted = cep.copy()
    cep_lifted[order + 1:] = 0.0
    # 4) FFT 回频率域 + exp
    log_env = np.fft.fft(cep_lifted).real[:K]
    env = np.exp(np.clip(log_env, -20, 20))
    # 5) 归一化到 mag 峰值
    env *= mag.max() / max(env.max(), 1e-12)
    return env
```

默认 order = `N//20`，对 N=1024 是 50。这个量级：
- 能跟踪 F1=500Hz（F1 周期 = 96 采样 > order=50 的临界 50）
- 不能跟踪更窄的共振峰（BW<50Hz），但实际语音共振峰 BW 通常 40-150Hz
- 性能：单帧 ~0.15ms（B=512 / N=1024），远低于块预算 10.67ms

### 10.4 测试（121 PASS / 0 FAIL，新增 7 项）

```
[1] test_lpc_envelope_positive_finite        — 9 种 K × order 组合无 NaN/Inf/非正
[2] test_lpc_envelope_smoother_than_max_filter — 一阶差分方差 < max_filter 的 50%
[3] test_lpc_envelope_correlates_with_true   — 与真实包络的归一化相关系数 > 0.70
[4] test_lpc_envelope_peak_locations         — 局部极大值命中 500/1500/2500 ± 200Hz
[5] test_envelope_method_selection           — 'lpc' 和 'max_filter' 两种都跑通
[6] test_default_envelope_method_is_lpc      — 默认值 = 'lpc'（v0.5.2 起）
[7] test_formant_corrector_lpc_does_not_crash_pipeline — VoicePipeline 全链路 +6 半音不死
```

### 10.5 向后兼容

新参数 `envelope_method="lpc"` 默认开启，**所有 v0.5.1 之前的用户都自动获得更平滑的共振峰校正**。想回到老实现只需传 `envelope_method="max_filter"`。实测 `max_filter` 仍可用且行为不变。

### 10.6 主观听感

因为 LPC 估的包络不再被谐波密度"撑高"，升 +6 半音后共振峰校正"拉回"的距离更精确：
- 男声 → 女声（semitones=+6, form_shift=1.0）：之前可能把共振峰拉到原位稍**低**的位置（因为 max_filter 估高了包络），现在精确回到原位。
- 女声 → 男声（semitones=-6, form_shift=1.0）：同样更精确。

---

## 11. Phase D：pitch envelope smoothing + 随机相位（v0.5.3）

### 11.1 背景

v0.5.2 落地后用户感觉「基本升降调精确了但还有数字感」。两件事：

1. **GUI 拖音高滑条时，ratio 是阶跃的** → 听感有「咔哒」声
2. **共振峰校正后偶有「颗粒感」** → 周期性的「嗡嗡」

两个症状听起来像同一回事（都是「不连续」），但根因不同，分别解决。

### 11.2 ratio 中值滤波（pitch envelope smoothing）

**问题分析**：当前 `process(x, ratio)` 每次调用 ratio 都是独立的常数。GUI 拖滑条时，假设用户从 1.0 拖到 1.5，每块 ratio = 1.5 是瞬时切换。听感上会出现明显的「频率跳变」。

**为什么不用线性插值**：中值滤波比线性插值更鲁棒 —— 如果用户的 ratio 因某些原因抖动（比如鼠标精度、卡顿），线性插值会继承抖动；中值会滤掉单点抖动。

**实现**：

```python
# StreamingPitchShifter.__init__ 加 ratio_smoothing_blocks 参数
self.ratio_smoothing_blocks = max(1, int(ratio_smoothing_blocks))
self._ratio_window: list[float] = []

# process 开头加：
win = self.ratio_smoothing_blocks
if win > 1:
    self._ratio_window.append(float(ratio))
    if len(self._ratio_window) > win:
        self._ratio_window = self._ratio_window[-win:]
    self.ratio = float(np.median(self._ratio_window))
else:
    self.ratio = float(ratio)
```

**窗口大小选择**：默认 `ratio_smoothing_blocks=1`（= 关闭，完全保留 v0.5.2 行为）。推荐值：
- GUI 平滑拖动：`5 ~ 10` 块（≈ 50~100 ms @ B=512）
- 突然大跳变（从 +0 半音到 +12 半音）：用户期望「立刻」变，所以**不**开启中值

测试见 `test_ratio_smoothing_smoothes_transitions`：当 ratio 从 1.0 突然变 1.5、win=5 时，前 2 块仍保持 1.0（窗口里 1.0 占多数），第 3 块起跳到 1.5。

### 11.3 随机相位（去梳状伪影）

**问题分析**：FormantCorrector 内部的 OLA（overlap-add）拼接是用 sqrt-Hann 窗的。如果所有帧的相位都是「自然连续」（每帧的相位都是前一块的延续），OLA 会在每个窗重叠处都形成同相相加 → 形成「整列同步」 → 频谱上出现间隔 `sr/B` 的梳状谱线（Comb filter）。

**为什么 WSOLA 输出本身没问题**：StreamingPitchShifter 用 WSOLA 保证输出帧的相位与输入相位是「自然连续」的（窗口只在「能量匹配」的位置取）—— 这是 WSOLA 的优势，但对 OLA 是劣势。

**解法**：给每帧所有频谱 bin 加同一个随机相位偏移（rad）。帧内相对关系不变（谐波结构不变），但帧间相位被打散 → OLA 不再「整列同步」。

**关键决策：把 phase_jitter 加在共振峰校正而不是 WSOLA**

为什么不在 WSOLA 阶段加？WSOLA 调的是「输出频率」（基频），相位必须稳定才能正确调音高；共振峰校正只调「包络形状」，相位只是「拼接相位」。

```python
# _process_frame 改相位应用：
phase = np.angle(F)
if self.phase_jitter > 0.0:
    phase = phase + self._rng.uniform(-self.phase_jitter, self.phase_jitter)
out = np.fft.irfft((new_mag * np.exp(1j * phase)).astype(np.complex64),
                   n=self.N).real.astype(np.float32)
```

**随机数发生器**：

```python
self._rng = np.random.default_rng(seed=0xA5C3 ^ id(self))
```

用 `id(self)` 加盐种子 → 不同实例独立；同一实例每次跑相位序列稳定（可重现）。

### 11.4 phase_jitter 默认值 0.10 的取舍

| 值 | 效果 |
|----|------|
| 0 | 严格保留 v0.5.2 行为，梳状伪影可能存在 |
| 0.05 ~ 0.10 | 推荐区间：轻微去梳状，主观听感明显改善 |
| 0.20 | 较强去梳状，但有「轻微金属感」（相位被打得太散） |
| 0.30 | 测试用：单块 1.70ms（性能 ok），但听感可能太「散」 |

最终选 `phase_jitter=0.10` —— 推荐区间中点。

### 11.5 测试（136 项 PASS）

新增 8 项测试：

1. **test_ratio_smoothing_basic** — 开启/关闭都能跑
2. **test_ratio_smoothing_smoothes_transitions** — 验证 [1.0]*10 + [1.5]*10 时中值序列 = [1.0, 1.0, 1.5, 1.5, 1.5]
3. **test_ratio_smoothing_window_size_1_is_no_op** — win=1 时严格保持输入 ratio
4. **test_phase_jitter_disabled** — phase_jitter=0 时两次跑严格相同（确定性的保证）
5. **test_phase_jitter_preserves_harmonic_structure** — phase_jitter > 0 后基频峰仍在原位（帧内谐波关系不变）
6. **test_phase_jitter_differs_from_disabled** — phase_jitter > 0 时两次跑出**不同**输出（随机相位起作用了）
7. **test_phase_jitter_per_block_latency** — 单块 1.70ms < 预算 8.53ms（80%）
8. (前面已有 Phase C 测试被覆盖)

总测试数：121 → 136 PASS（+15 项涵盖 Phase A/B/C/D）。其中 Phase D 直接新增 8 项。

### 11.6 向后兼容

- `ratio_smoothing_blocks=1`（默认）= 完全与 v0.5.2 一致
- `phase_jitter=0`（如果外部传）= 完全与 v0.5.2 一致
- VoicePipeline / process_offline 的旧调用方继续工作

### 11.7 主观听感（开发者视角，未跑用户 A/B 测试）

- **GUI 拖滑条**：从「咔哒」变成「柔和过渡」，符合用户期望
- **梳状伪影**：phase_jitter=0.10 在男声 +0 半音下基本听不出「嗡嗡」；在女声 +6 半音、谐波密时仍能听出轻微改善

### 11.8 还可以做（Phase E 候选）

- **PLL（锁相环）**：跟踪实时基频，避免 wsola 在 F0 抖动时对齐错位
- **自相位随机化**：每帧的 jitter 大小根据信号能量自适应（安静时更小，响亮时更大）
- **预畸变（pre-emphasis）**：先做 `x' = x - 0.97*x[-1]` 再处理，恢复时再做相反操作，让高频更突出

---

## 7. 还可以做（v0.4+ 候选）

- 用 LPC（线性预测）估包络代替 rolling max —— pprablanc 的半成品仓库
- 让 form_shift_ratio 跟着 semitones 自动适配（默认 +3 半音就配 +1.05 form_shift）
- 给 VOICE_PRESETS 每项带附加音效配方（per-preset echo_ms delay 40ms/350ms、tremolo、vibrato、reverb）
  —— 当前 EffectChain 只有 echo_ms=130 默认、phone、robot 三种，要支持 SUONSUN9527 的多参音效需要先扩 EffectChain
- 实时声码器跑 RVC 轻量版（如果用户能提供目标音色录音）

---

## 12. v0.8 / v0.9：从「实时变声」到「文字转语音」

### 12.1 为什么又转了一次方向

时间线上的三次转向，每次都是同一个原因：**用户要的「像真人」不是 DSP 能给的**。

| 版本 | 路线 | 结果 |
|---|---|---|
| v0.1 - v0.5.3 | DSP 实时变声（WSOLA + 共振峰 + 16 效果器） | 音准没问题，但**不像真人**（用户 m01187：「电音和杂音」）→ v0.6.0 回退到 v0.2.0 干净基线 |
| v0.6 - v0.7.1 | RVC 神经声码（预训练角色模型） | 能克隆音色，但**本机跑不动**：AMD 780M 512MB 无 CUDA，CPU 推理 ~750ms 延迟，且实时场景下延迟不可接受 |
| v0.8 | IndexTTS-2.5 本地克隆 | 离线可用，但**加载期撞内存提交上限**，长文本任务做不了 |

用户 m01897 明确说「改用 tts」，于是把实时麦克风整条链路拆掉，改成
「输入文字 → 输出指定音色的语音」。**离线合成没有延迟约束**，这让
「等 30 秒换一个好音色」变成可接受的交易。

### 12.2 v0.8.0：IndexTTS-2.5 本地引擎

选型对比（详见 `docs/MINIMAX_TTS_API.md` 同级的调研记录）：

- IndexTTS-2.5：0.8B、5 语言、零样本克隆、情感/语速可控、单模型包
  → 选它是因为**不用拼凑**（cnhubert / BERT / SV / SoVITS base / GPT base 各下各的）
- GPT-SoVITS v2ProPlus：要 6 个组件、还要 `config.py`（上游仓库里被 gitignore 掉了），
  且 pip 装 37 个依赖在 Tsinghua 源上反复卡死 → **放弃**
- Edge-TTS / pyttsx3：质量够但没有「角色音色」概念，跟 30 预设对不上

工程实现：

- `indextts_engine.py` — 封装 `IndexTTS2`，含低内存加载的两处缓解：
  - `_enable_mmap_torch_load()`：`torch.load` 对 ≥64MB 的 `.pth/.ckpt` 走 `mmap=True`。
    原因：`gpt.pth` 3.1 GB，默认加载会反序列化出一整份副本再拷进模型，
    峰值多 3.1 GB 提交量；实测从崩溃变成 0.6 s 成功。
  - `_ascii_alias()`：把含中文的路径换成指向同目标的 ASCII junction（`D:\vc`）。
    原因：torch 的 mmap 走 `CreateFileMapping`，在中文路径上有「映射建好但指针不可用」
    的踩坑表现，`load_state_dict` 里 `copy_` 直接 ACCESS_VIOLATION。
    实现要点：`os.path.splitdrive(path)[0] + os.sep` —— **必须补 `os.sep`**，
    否则 `splitdrive` 返回 `"D:"`，扫的是错的目录。
- `indextts_server.py` — HTTP 服务（`/health`、`/tts`），`faulthandler.enable(all_threads=True)`
  用来抓加载期的原生崩溃栈
- `indextts_worker.py` — GUI 进程（3.12）↔ 推理进程（3.11）之间的 JSON-Lines 通道

### 12.3 吞尾音问题的定位（v0.9.0 前置修复）

用户反馈「发送语音消息时会吞掉尾音」。实测 4 段不同长度的合成：

| 时长 | 末 100ms RMS | 末 100ms Peak | 尾静音 |
|---|---|---|---|
| 15.4s | 12.9 | 186 | 143 ms |
| 3.95s | 239 | 1797 | 80.7 ms |
| 2.43s | 255 | 2251 | 85.0 ms |
| 1.23s | 3.6 | 12 | 497 ms |

50ms 窗口细查 3.95s 那条：`t-1500~t-1300` 前句残音 → `t-1300~t-1150` 静音 →
`t-1100~t-150` **持续强语音**（RMS 2000-10000）→ `t-50ms` 突然静音。

根因不在后处理，而在**上游模型**：

- IndexTTS 的 `target_lengths = code_lens * 1.72`，`code_lens` 由 `stop_mel_token`
  随机采样决定。停止符落得偏早 → mel 帧提前截止 → 最后一个采样点离零很远
  → 一阶差分在文件末尾出现一个巨大的 step（值 5242 > 5000 判 click）
- BigVGAN 在 mel 边界处感受野不完整，可能再叠一层突发噪声
- 上游 `utils/common.py` 其实有 `fade_out_tail`（20ms 升余弦），但 `infer_v2_5.py`
  走的是 `save_pcm_wav()`，**没调它**

修法：在 `indextts_engine.synthesize` 里，推理结束后读回音频，自己做
**20ms 升余弦淡出 + 250ms 静音 padding**。升余弦的好处是「到零时斜率也为零」，
不会引入新的不连续点。

验证（同一段文本）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 末 100ms Peak / RMS | 1797 / 239 | **0 / 0** |
| 末 50ms Peak / RMS | 强语音 | **0 / 0** |
| 尾静音 | 80.7 ms | **361.2 ms** |
| 末活跃点 | t-50ms（被切） | **t-4.08s** |
| step_max 21557 位置 | 含文件末尾 | **全在中段**（真实音素过零） |

教训：**排查「声音被截断」要先区分「模型没生成」和「生成完但被切」**。
前者看时长，后者看末尾波形 —— 一阶差分的 step 是不是孤立在文件边缘，
是最快的判据。

### 12.4 v0.9.0：MiniMax 云端后端

用户 m04147 要求「查看 minimax 提供的 api 调用 tts，整理出来给我」。
产出 `docs/MINIMAX_TTS_API.md`（1003 行，17 章），把 T2A v2 的
同步 HTTP / WS / 双向流式 / 异步 / 克隆 / 声音设计 / 音色管理全部梳理清楚。

然后实现 `minimax_tts.py`。几个刻意的决定：

**1）纯标准库，零第三方依赖。**
这台机器上 `requests` / `websockets` 有，但 `soundfile` / `sounddevice` 之前是缺的
（现已装进系统 Python 3.12）。MiniMax 客户端只用 `urllib.request` + `json`，
保证在任何 Python 3.8+ 环境都能直接跑。WebSocket 那两个方法做成了
「有 `websockets` 就用，没有就抛明确错误」，而不是硬依赖。

**2）multipart 手工拼。**
标准库没有 multipart 编码器，所以 `_post_multipart()` 自己拼 boundary
（`----MiniMaxBoundary` + uuid4hex），字段顺序是 `purpose` 在前、文件在后。

**3）30 个角色预设一对一映射。**
`MINIMAX_PRESETS` 严格对齐原项目 `voice_core.VOICE_PRESETS` 的 30 个角色名和分组。
写完后跑测试发现漏了「御姐」，而且「外星人 / 电音女王 / 兽人」被我错分到「女」组
（它们在原项目里是「特效」组）。**最后是去 git 里 `git show f6e8850:voice_core.py`
把原始列表抠出来对了一遍才改对** —— 手抄列表必然出错，以 git 历史为准。

每个预设不只是换一个 `voice_id`，还带 `speed` / `vol` / `pitch` / `emotion` /
`voice_modify`（后者的 `sound_effects` 有 4 个官方取值：spacious_echo /
auditorium_echo / lofi_telephone / robotic，正好覆盖「幽灵 / 回声 / 电话音 / 机器人」）。

**4）异步接口的字段名不一样。**
`/v1/t2a_async_v2` 的 `audio_setting` 里采样率叫 `audio_sample_rate`，
不是同步接口的 `sample_rate`。`async_create()` 里自动做了转换。
这类「同一个概念两个接口两个字段名」的坑，靠测试固定住最省事。

**5）Tk 不是线程安全的。**
合成请求放在后台线程，但**所有控件的值必须在 GUI 线程里读出来再传进去**。
第一版在 worker 里直接 `self.cb_mm_lang.get()`，虽然大多数时候能跑，
但这是定时炸弹 —— 现在改成 GUI 线程读值 → 构造 client → 传参给 worker。

### 12.5 顺手解决的老问题：内存

v0.8.0 那次 IndexTTS 加载崩溃，根因查清楚了：

- 物理内存 15.2 GB，可用 **0 MB**；提交 34.8 GB / 上限 40.5 GB
  （`AutomaticManagedPagefile=False`，页面文件手工固定 24.8 GB）
- 大户：`StarRail` 游戏 priv=5,126 MB（ws 只有 364 MB，大部分已换出）、
  WorkBuddy ×2、微信、B站
- 引擎稳态需要 ~7.5 GB：GPT fp32 3.1 + w2v-bert fp32 2.2 + BERT 0.65 +
  semantic_codec 0.75 + s2mel 0.4 + bigvgan 0.43
- 上游 `infer_v2_5.py:100-120` 在 CPU 上**强制 `use_bf16 = False`**
  （`if device is not None: self.use_bf16 = False if device == "cpu" else use_bf16`；
  不传 device 则走 else 分支同样置 False）→ GPT 实际是 fp32

对照实验（`_probe_mem2.py`）很干脆：GPT 驻留时加载 w2v-bert → 剩 799 MB → AV；
GPT `del` 释放后加载 → 剩 4599 MB → 成功。**不是代码 bug，是内存提交量耗尽。**

用户明确拒绝了三项「治本」方案（关 StarRail / 扩页面文件 / 减少常驻软件）
和「服务开机自启」。既然不能动系统，就只能从代码侧挤：

- `torch.load` 走 `mmap=True`（省 3.1 GB 副本）
- `semantic_model` 转 bf16（省约 1.1 GB），需要配套把 forward 的输入转 bf16、
  输出 hidden_states 转回 fp32，否则报 `mixed dtype (CPU): expect parameter to
  have scalar type of Float`
- `gc.collect()` + `EmptyWorkingSet(-1)` 归还碎页

**这三条都没能彻底解决 —— 加载期仍会撞上限。** 所以 v0.9.0 的产品决策是：

> **默认后端改成 MiniMax 云端。IndexTTS 改成「选中后才懒加载」。**

这一改顺带解决了另一个体验问题：以前 GUI 一启动就拉起 IndexTTS worker，
加载 30-70 秒、吃 7.5 GB，用户想用云端也躲不掉。现在 `tts_app.py` 默认
后端是 MiniMax，`IndexTTS worker not autostarted` 也被写成了测试用例。

### 12.6 测试

- `test_minimax_tts.py` — 40 项，全部用本地 `ThreadingHTTPServer` mock，
  **不发外网请求、不需要 API Key**。覆盖：请求体构造（bitrate 只在 mp3 时出现、
  可选块的有无）、20 个官方错误码的解析、hex/url 音频提取、multipart 的
  boundary 与字段顺序、异步接口的 `audio_sample_rate`、克隆 voice_id 的
  4 类校验、30 个预设与原项目的一致性（含分组）
- `test_tts_app.py` — 9 项 GUI 冒烟：后端切换的面板显隐、角色下拉 30 项、
  预设反查、缺 Key 时抛错、**worker 不自动启动**
- 合计 **49 PASS / 0 FAIL**

一个环境上的坑：GUI 测试里判断「面板有没有显示」不能用 `winfo_ismapped()` ——
测试时窗口是 `withdraw()` 的，所有子控件的 `ismapped` 恒为 0。
改用 `winfo_manager()`（返回布局管理名，没被布局则为空串）才对。

### 12.7 还想做的

- **MiniMax 声音克隆接进 GUI**：克隆音色 7 天不用会被系统删，得有个列表管理
- **双向流式接进 GUI**：服务端自动断句 + 逐句返回，比一次性合成体感好
- **IndexTTS 的 bf16 彻底化**：绕开上游 CPU 强制 fp32 的逻辑
  （monkeypatch 或传 `device="cpu:0"`），把稳态需求压到 4-5 GB
- **30 个预设的 A/B 主观听感**：目前映射是按参数直觉配的，没有真实试听对比

## 13. v0.9.1：拿到真 Key 之后的第一次真实调用

前面 12.4 的 MiniMax 客户端全部是**对着 mock server 写的**。用户终于把订阅的
Key 给了出来（2026-10-31），第一件事就是拿真接口跑一遍 —— 结果**第一把就挂了**。

### 13.1 Key 与域名是绑定的（本次最大的发现）

```
POST https://api.minimax.io/v1/get_voice
  -> {"base_resp":{"status_code":2049,"status_msg":"invalid api key"}}
```

同一个 Key、同一个请求体，换个域名就完全正常：

```
POST https://api.minimaxi.com/v1/get_voice
  -> {"system_voice":[{...}], "base_resp":{"status_code":0}}
```

一开始以为是 Key 过期 / 复制时少了字符 / 鉴权头写错。排查方式是把**三个域名 ×
三个端点**（`get_voice` / `t2a_v2` / `chatcompletion_v2`）全打一遍：

| 域名 | get_voice | t2a_v2 | chatcompletion_v2 |
|---|---|---|---|
| `api.minimax.io` | 2049 | 2049 | 2049 |
| `api.minimaxi.com` | ✅ | ✅ | ✅ |
| `api.minimax.chat` | ✅ | ✅ | ✅ |

三个端点全 2049 → 说明是**域名层面的 Key 命名空间**，不是某个接口的问题。
MiniMax 的 `.io`（新平台）和 `.com`（老平台）用的是两套互不相通的 Key。

**这意味着写死 `https://api.minimax.io` 的客户端，遇到 `.com` 的 Key 就完全不可用。**
所以改成：默认 `base_url="auto"`，按 `api.minimaxi.com` → `api.minimax.io` →
`api.minimax.chat` 的顺序试，**撞到 2049 就自动换下一个域名重试**，并把成功
的那个域名记在 `client.base_url` 上。显式指定域名时不回退（用户既然钉死了就
尊重他的选择，错误原样抛出）。

### 13.2 回退逻辑要覆盖 4 条路径

不只是同步 HTTP。抽了 `_try_hosts(fn, what)` 包住 `_post_json` / `_get_json` /
`_post_multipart`；流式和 WebSocket 因为是自己管理连接，得单独包：

| 方法 | 传输 | 2049 怎么出现 | 怎么回退 |
|---|---|---|---|
| `synth` / `list_voices` / `clone_voice` … | HTTP | `base_resp.status_code` | `_try_hosts` 统一处理 |
| `synth_stream` | HTTP SSE | 流**内部**的 `base_resp` | 外层 while 循环重建连接 |
| `synth_ws` | WebSocket | `task_start` 事件里带 2049 | `_run_for(host)` 逐个 host 试 |
| `synth_bidi_ws` | WebSocket | 同上 | 同上 |

顺带修了个坑：WebSocket 的 `task_start` 失败分支原本**没带 code**，
`raise MiniMaxError(f"task_start 失败: {started}")` —— 没有 code 就没法判断
是不是 2049，也就没法回退。补上 `code=br.get("status_code")` 才让 WS 路径能回退。

### 13.3 30 个角色预设的真实扫描

改完客户端，对 30 个预设各发一次真实请求（`speech-2.8-turbo`，wav / 32 kHz / 单声道）：

| 组 | 角色数 | 成功 | 失败 |
|---|---|---|---|
| 原 | 1 | 1 | 0 |
| 女 | 8 | 8 | 0 |
| 男 | 7 | 7 | 0 |
| 童 | 7 | 7 | 0 |
| 特效 | 7 | 7 | 0 |
| **合计** | **30** | **30** | **0** |

耗时 0.78 ~ 2.02 s（平均 1.09 s），产物 32 kHz 单声道 WAV。
`Robot_Armor`（机器人）、`spacious_echo`（幽灵）、`auditorium_echo`（回声）、
`lofi_telephone`（电话音）这四个用了 `voice_modify.sound_effects` 的也都成了。

**但也暴露了一个假阳性**：之前用官方文档整理的音色表（`.io` 那份）校验过一遍
「30 个 voice_id 全部命中」，结论是"没问题"。实际上 `.com` 平台的音色表是
**另一套**，只是恰好包含了全部 30 个 id —— 属于运气，不是验证。

### 13.4 顺带做的两件事

- **`synth(..., with_meta=True)`**：返回 `(音频字节, 完整响应)`，能拿到
  `extra_info.usage_characters` 等计费信息，方便算成本。
  顺手发现官方报的是 `usage_characters`（按 Unicode 码点算，10 字的中文报 18）
  和 `word_count`（10）两个数，计费应该看前者。
- **GUI 加「域名」下拉框**：默认「自动探测」，也可以手动钉死一个。
  「测试连接」会在日志里打印**实际生效的域名**，免得用户对着 2049 干瞪眼。

### 13.5 这次踩的测试环境的坑

`test_missing_api_key_raises` 在加了「启动时把配置里的 Key 填进输入框」之后
**突然挂了** —— 因为开发机上 `%APPDATA%/voice-changer/minimax.json` 里已经有
真 Key，`_load_mm_key()` 把它填进了 entry，测试只 monkeypatch 了
`CONFIG_PATH` 却没清 entry。

这不是测试写错了，是**测试环境污染了**：`CONFIG_PATH` 指向真实用户目录，
而产品代码也会去读它。修法是测试里把 entry 也清掉。更彻底的做法是把
`CONFIG_PATH` 指到 tmp_path —— 已经这么做了，只是 entry 那条路径也要一起清。

### 13.6 回归测试

| 文件 | 项数 | 结果 |
|---|---|---|
| `test_minimax_tts.py` | 47（原 40 + 新 7） | PASS |
| `test_tts_app.py` | 9 | PASS |
| **合计** | **56** | **56 PASS / 0 FAIL** |

新增 7 项里有 4 项是**专门测域名回退**的（需要第二个「永远返回 2049」的
mock server）：

- `test_default_base_url_is_auto`
- `test_base_urls_cover_minimaxi_and_io`
- `test_auto_host_falls_back_to_next_host_on_2049`
- `test_explicit_base_url_does_not_fall_back`
- `test_all_hosts_bad_raises_2049`
- `test_synth_with_meta_returns_response`
- `test_synth_without_meta_returns_bytes`

---

## 14. 发布会话里的网络 / git / 编码踩坑

v0.9.1 收尾时把发布流程重跑了一遍，踩到的坑都记在这里。

### 14.1 github.com 的 geo-DNS 答案在这台机器上不通

| 主机 | DNS 答案 | 443 实测 |
|---|---|---|
| `github.com` | 20.205.243.166 | **TCP 超时** |
| `api.github.com` | 20.205.243.168 | 正常（`/rate_limit` → 200）|
| `uploads.github.com` | 20.205.243.161 | 正常（POST asset → 201）|

症状：`git push` 报
`fatal: unable to access '...': Failed to connect to github.com port 443 after 21084 ms`。

解法（git 2.53 支持 `http.curloptResolve`）：

```bat
git -C D:\代码\voice_changer ^
    -c "http.curloptResolve=github.com:443:140.82.112.3" push origin main
```

**web 主机和 API 主机必须用两套地址。** 我一开始图省事，把
`api.github.com` / `uploads.github.com` 也一起 remap 到 `140.82.112.5`，
结果 `POST /releases/{id}/assets` 一律返回 JSON 的 `404 Not Found`：

- `140.82.112.3` 对 `api.github.com` 返回 `301 → https://github.com/...`（web 边缘）
- `140.82.112.5` 对 `api.github.com` 返回 200（API 边缘），但**没有实现 asset 上传路由**
- `uploads.github.com` 用**普通 DNS**（20.205.243.161）直接 POST → `HTTP 201`

所以最终方案是：**git 走 curloptResolve，API / 上传走普通 DNS**。

### 14.2 `D:\代码` 和 `D:\代码\voice_changer` 是两个 git 仓库

- 父仓库 `D:/代码`：**一个 commit 都没有**（branch `master`，无 remote），里面混着几十个别的项目（cablecar / qisheng-chess / ElytraRing / sanxiang / elytra_* …）
- 子仓库 `D:/代码/voice_changer`：本项目，remote = `github.com/LuoGu14514/voice-changer`

`git status` 一定要带 `-C "D:\代码\voice_changer"`。
我一度把 `_live_sweep/` 写进了**父仓库**的 `.gitignore`（9 行 → 10 行）。

### 14.3 PowerShell 控制台是 GBK，读 UTF-8 会吃掉换行

`Get-Content` 读 UTF-8 中文文件时乱码，而且**会把换行吃掉** —— GBK 双字节解码把
`0x0A` 当成某个字的第二个字节吃掉了，看起来像两行并成一行：

```
# 运行时写的日志?rtc.log        <- 实际文件里是正确换行，终端显示成一行
```

改文件一律用：

```powershell
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($p, $txt, $enc)
```

同理，`curl -d '{"name":"…中文…"}'` 也会被控制台编码搞坏，GitHub 回
`400 Problems parsing JSON`。参数里有中文时走 Python + urllib，别走 curl。

### 14.4 `git archive` 打包源码 zip

```bat
git archive --format=zip --prefix=voice-changer-v0.9.1/ -o voice-changer-v0.9.1-source.zip HEAD
```

只含 **tracked** 文件，12 GB 的模型 / `_indextts_venv` / `models/` 天然不会泄漏。
v0.9.1 zip = 225127 bytes。

### 14.5 v0.9.1 发布结果

| 项 | 值 |
|---|---|
| commit | `4604736` |
| tag | `v0.9.1` |
| release id | `429924617` |
| asset id | `695338457`（225127 bytes）|
| 回归 | `test_minimax_tts.py` 47 + `test_tts_app.py` 9 + `test_voice_core.py` 14 = **70 PASS / 0 FAIL** |


## 15. v0.10.0：「自定义角色」—— 三条路和它们的边界

### 15.1 问题的由来

用户在 v0.9.1 交付后问了一句：**「不可以自定义角色吗」**。

之前 30 个角色是「原项目 v0.4.0 的角色表 → MiniMax 官方系统音色的手工映射」，
每一项都是写死在 `MINIMAX_PRESETS` 里的。用户想要的是**自己造角色**的能力。

### 15.2 先探测权限，而不是先写代码

拿手上的订阅 Key 逐个打接口，得到一组决定性的事实：

| 接口 | 结果 | 含义 |
|---|---|---|
| `/v1/get_voice`（voice_cloning） | 账号里已有 **1 个**克隆音色 `NahidaDesktopTest_20260812_29bc0d2a` | 之前建过 |
| `/v1/get_voice`（voice_generation） | 0 个 | 没建过设计音色 |
| `/v1/voice_clone` | **2061** `your current token plan not support model, voice_clone` | **套餐不支持新建克隆** |
| `/v1/voice_design` | **2061** `... not support model, voice_design` | **套餐不支持新建设计** |
| `/v1/t2a_v2`（用已有克隆音色） | OK，50676 bytes / 3060 ms | **已有的自定义音色照常能用** |
| 8 个 TTS 模型 | 全部可用 | 合成能力本身没问题 |

结论很清楚：**这把订阅能「用」自定义音色，不能「造」新的自定义音色**。
所以「自定义角色」必须拆成三条路，而不是一条。

### 15.3 三条路的分工

| 方式 | 需要什么 | 得到什么 | 成本 | 本机可行性 |
|---|---|---|---|---|
| **零样本参考音频** | 一段 3-15 秒干净语音 | 一个新角色，音色来自这段音频 | 本地跑 | **可行**（已实测跑通） |
| **拉取云端自定义音色** | 账号里已有的克隆/设计音色 | 把它们加进角色库 | 免费 | **可行**（已实测跑通） |
| **声音克隆 / 声音设计** | 支持该能力的套餐 | 云端新建音色 | 按 MiniMax 规则计费 | **被 2061 挡住**（但保留入口 + 明确提示） |

第三条虽然实测被挡住，**入口还是要留着**：换个套餐就能直接用，
而且界面上要把 2061 的原因说清楚，不能只丢一个错误码给用户。

### 15.4 IndexTTS 零样本：唯一真正「零训练」的路

IndexTTS 的 `spk_audio_prompt` 本来就是给零样本克隆设计的——引擎侧一直支持
传任意 `ref_audio`，缺的只是 GUI 上一个「选文件」的入口。

所以这条路的关键不是写新算法，而是**把参考音频变成一个可保存、可复用的角色**。

顺带解决了一个更硬的问题：**本机 IndexTTS 加载一直会崩**。

### 15.5 一个意外收获：semantic bf16 才是崩溃的真凶

§12 记过一次排查：IndexTTS 加载阶段 `exit=-1073741819`（ACCESS_VIOLATION），
崩在 `transformers` 的 `_load_state_dict_into_meta_model`，
先后排除了 safetensors 版本、deepspeed 缺失、模型损坏、权限告警，
最后定性为「**提交量耗尽**」——语义模型 w2v-bert 在 fp32 下要 ~2.2 GB。

当时做了 `_enable_bf16_semantic_model()` + `_patch_semantic_forward()`（进 bf16、
出 fp32），但**因为更早阶段就崩了，一直没能验证**。

这次重试：

| 条件 | 结果 |
|---|---|
| `semantic_bf16=False`（fp32），threads 默认 | `exit=-1073741819`（AV，一如既往） |
| `semantic_bf16=True`，`threads=4` | **exit=0**，加载成功，合成成功 |

commit 额度当时是 7423 MB 余量，引擎 fp32 稳态要 ~7.5 GB —— 就差那么一点，
fp32 正好卡在悬崖边上；bf16 把语义模型从 2.2 GB 压到 ~1.1 GB，就过去了。

于是把 `semantic_bf16` 的默认值从 `False` **翻成 `True`**
（`indextts_engine.py` / `indextts_server.py` 同步），命令行开关从
`--semantic-bf16` 改成 `--no-semantic-bf16`（默认开，要退回得显式关）。

> 这也解释了为什么「同样的代码有时能跑有时崩」：不是随机，是提交量的函数。

### 15.6 worker 和 engine 重复实现 → 顺手修掉的两个 bug

`indextts_worker.py`（GUI 走的那条路）和 `indextts_server.py` 各自写了一份
`IndexTTS2` 的加载代码，而 `indextts_engine.py` 里已经有封装好的
`IndexTTSEngine`。三份实现 → 三种行为：

| 能力 | engine | server | worker |
|---|---|---|---|
| `torch.load(mmap=True)` | ✓ | ✓ | ✗ |
| ASCII 路径别名 | ✓ | ✓ | ✗ |
| 语义模型 bf16 | ✓ | ✓ | ✗ |
| **末尾淡出（修吞尾音）** | ✓ | ✓ | **✗** |

最后一条是**真 bug**：§12 那次「吞尾音」的修复（20ms 升余弦淡出 + 250ms 静音）
只加在 `indextts_engine.py` 上，**GUI 实际用的 worker 根本没走那段代码**——
所以 GUI 里 IndexTTS 合成出来的东西**尾音还是会被切**。

这次把 worker 改成直接用 `IndexTTSEngine`，四行能力一次性对齐，
重复实现也少了一份。

> 教训：一个修复如果只落在「被测到的那条路径」上，用户实际用的那条路径等于没修。

### 15.7 本地角色库设计

角色不能只活在内存里，要能存、能改、能导出。选了个纯 JSON：

```
%APPDATA%oice-changer\custom_voices.json
{
  "version": 1,
  "voices": [
    {"name": "我的角色", "group": "自定义", "backend": "indextts",
     "voice_id": "", "ref_audio": "D:\音频\我的角色.wav",
     "speed": 1.0, "pitch": 0, "emotion": "",
     "modify": null, "source": "local", "note": "零样本克隆",
     "created": "2026-10-31 20:00:00"}
  ]
}
```

几个决定：

1. **`backend` 字段决定它出现在哪个后端的角色列表里**。
   `indextts`（有 `ref_audio`）走本地零样本，`minimax`（有 `voice_id`）走云端。
   混在一起会出「没有云端音色 ID 却发给 MiniMax」的错，所以 GUI 侧按 backend
   分流 + 明确拦截。
2. **`get_preset()` 统一查系统预设和自定义角色**，GUI 侧不用分两套取数逻辑。
   但 `grouped_presets()` 只在**真的有自定义角色时**才追加「自定义」组，
   否则 30 个系统预设的分组断言会被打破。
3. **写文件用 `os.replace()` 原子替换**，避免中途崩溃留下半个 JSON。
4. 读文件时容错：JSON 坏了就当空列表，不让 GUI 起不来。

### 15.8 顺手修的 upload_file bug

实测 MiniMax `/v1/files/upload` 的响应是**嵌套**的：

```json
{"file": {"file_id": 458331960050266, "bytes": 470332, "filename": "...", "purpose": "voice_clone"},
 "base_resp": {"status_code": 0, "status_msg": "success"}}
```

而原实现取的是顶层的 `obj.get("file_id")` → 拿到 `None` →
抛「上传成功但没拿到 file_id」。原来的 mock 测试也是按顶层写的，
所以**测试全绿但真实调用必挂**。

修法是嵌套优先、顶层兜底：

```python
file_obj = obj.get("file") or {}
file_id = file_obj.get("file_id", obj.get("file_id"))
```

并补了一条用真实响应结构的新用例。

> 教训：mock 的响应结构得按**真实**的写，不能照着代码的期望写。
> 这条 bug 之所以能活这么久，就是因为测试数据和实现共享同一个误解。

### 15.9 下拉框里带「·」的角色名

角色下拉的显示格式是「分组 · 名字」，取名字时原来用
`label.split("·")[-1]` —— 角色名里自己带「·」就取错了。

改成按已知分组前缀剥离：

```python
for g in list(PRESET_GROUPS) + [CUSTOM_GROUP, "其它"]:
    prefix = g + " · "
    if label.startswith(prefix):
        return label[len(prefix):].strip()
```

另外 Windows 文件名保留字符（`< > : " / \ | ? *`）在自定义角色名下会炸，
补了 `safe_filename()` 清洗后再拼输出路径。

### 15.10 测试

| 文件 | 用例数 |
|---|---|
| `test_minimax_tts.py` | 57（新增 10：嵌套 file_id、角色库读写/校验/覆盖、分组追加、merged、坏文件容错、云端音色列举） |
| `test_tts_app.py` | 17（新增 8：自定义面板控件、IndexTTS 下拉出现自定义角色、`_current_ref` 取自定义音频、MiniMax 下拉出现自定义组、后端错配拦截、Treeview 列表、`name_from_label`、`safe_filename`） |
| `test_voice_core.py` | 14 |
| **合计** | **88 PASS / 0 FAIL** |

关键一点：GUI 测试要把 `minimax_tts.CUSTOM_VOICES_PATH` 指到临时文件
（`monkeypatch` + module 级 `MonkeyPatch`），**否则测试会读用户真实的
`%APPDATA%`，用户在 GUI 里加过角色就会让「角色下拉有 30 项」的断言炸掉**。

### 15.11 实测数据

**MiniMax 自定义角色**（用账号里已有的克隆音色）：

```
云端自定义音色: [('NahidaDesktopTest_20260812_29bc0d2a', '克隆')]
已加入角色库: 娜希达测试 → NahidaDesktopTest_20260812_29bc0d2a
合成 OK: 213674 bytes  1.06s  时长 3.32s
分组: [('原',1), ('女',8), ('男',7), ('童',7), ('特效',7), ('自定义',1)]
```

**IndexTTS 零样本自定义角色**（`indextts_2faff91361e04de29cd87dd0a3392fb4.wav`
作参考音频，`semantic_bf16=True`, `threads=4`）：

```
load 成功 → 合成成功
Total inference time: 141.94 seconds
Generated audio length: 5.78 seconds
输出: 6.03s / 22050 Hz / 单声道 / 16-bit / 266044 bytes
peak 0.950  rms 0.1722
尾 100ms peak = 0.0000   ← 淡出生效，没有咔哒声
```

RTF 23.5（CPU 纯推理，5.8 秒音频要 142 秒）—— 本地后端是「准离线」，
日常还是 MiniMax 云端顺手。

### 15.12 已知限制

- MiniMax 新建克隆/设计音色被套餐挡住（2061）。**已建的音色照常用**。
- IndexTTS 本地后端吃内存，本机提交余量紧张时仍可能加载失败；
  `semantic_bf16=True` 把崩溃概率从「基本必崩」降到「基本能成」，但根因
  （提交上限 40.5 GB、游戏 + IM 常驻）没变。
- 本地后端 RTF ~23，慢到不适合实时；实时需求还得靠 MiniMax。
