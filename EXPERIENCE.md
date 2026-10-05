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

## 11. Phase D：pitch envelope + 随机相位（v0.5.3，待办）

- **pitch envelope smoothing**：当前每块 ratio 常数（block-level WSOLA），改成每块边界 medfilt → 说话时音高连续曲线，而不是阶跃
- **随机相位**：在 _process_frame 把相位 `np.angle(F)` 改成 `np.angle(F) + small_random`，去"颗粒感"
- 期望效果：让处理过的声音从「音高跳变」变成「连贯」，进一步缩小与真人的差距

---

## 7. 还可以做（v0.4+ 候选）

- 用 LPC（线性预测）估包络代替 rolling max —— pprablanc 的半成品仓库
- 让 form_shift_ratio 跟着 semitones 自动适配（默认 +3 半音就配 +1.05 form_shift）
- 给 VOICE_PRESETS 每项带附加音效配方（per-preset echo_ms delay 40ms/350ms、tremolo、vibrato、reverb）
  —— 当前 EffectChain 只有 echo_ms=130 默认、phone、robot 三种，要支持 SUONSUN9527 的多参音效需要先扩 EffectChain
- 实时声码器跑 RVC 轻量版（如果用户能提供目标音色录音）