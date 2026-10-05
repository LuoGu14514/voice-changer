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

## 7. 还可以做（v0.4+ 候选）

- 用 LPC（线性预测）估包络代替 rolling max —— pprablanc 的半成品仓库
- 让 form_shift_ratio 跟着 semitones 自动适配（默认 +3 半音就配 +1.05 form_shift）
- 给 VOICE_PRESETS 每项带附加音效配方（per-preset echo_ms delay 40ms/350ms、tremolo、vibrato、reverb）
  —— 当前 EffectChain 只有 echo_ms=130 默认、phone、robot 三种，要支持 SUONSUN9527 的多参音效需要先扩 EffectChain
- 实时声码器跑 RVC 轻量版（如果用户能提供目标音色录音）