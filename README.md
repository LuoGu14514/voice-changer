# 简单变声器（实时麦克风变声）

一个只用 Python + numpy 写的实时变声小工具：**变调**（±12 半音）+ **共振峰校正** + **16 种效果器可任意组合** + **微抖动（jitter/shimmer）**，图形界面开箱即用。
算法是自写的「颗粒重叠相加 + WSOLA 相位对齐 + 谱包络分离」，不依赖任何音频处理库。

```
voice_core.py      DSP 核心（纯 numpy，可单独用）
voice_changer.py   图形界面 + 命令行
test_voice_core.py 纯数值自测（不需要声卡）
EXPERIENCE.md      算法研判 + 同类项目调研结论
启动变声器.bat     双击即用（无控制台窗口）
调试运行.bat       出问题时用这个启动，能看到报错
requirements.txt   numpy / sounddevice / soundfile
```

## 一、怎么用

1. 双击 **`启动变声器.bat`**（首次启动约 1~2 秒）。
2. **麦克风**：选你说话的麦克风。
3. **输出到**：想自己听效果就选耳机/扬声器；想给游戏、QQ、微信用就选 `CABLE Input`（见第三节）。
4. 点 **▶ 开始变声**，然后说话。想听效果可以戴耳机，避免麦克风把扬声器的声音收进去产生啸叫。
5. **音调**滑块拖到 ±12 半音；**共振峰位移**滑块（0.70~1.30）拖到想要的位置；或直接点预设按钮：
   **30 个内置预设**（详见下方「预设一览」）。
6. 「**启用共振峰校正**」勾上后，升/降调会自动把共振峰拉回原位 ——「男声 +3 半音」听感上
   是「还是男声、但变年轻了」，而不是「变成小孩音」。
7. **开始录音** → 说话 → **保存录音…** 可以导出变声后的 wav。

界面还会显示：实时电平条、是否削波、xrun（音频卡顿丢帧）计数、算法延迟估算。

命令行处理文件（可选）：

```bat
.venv\Scripts\python.exe voice_changer.py --file 输入.wav -o 输出.wav --semitones 5 --formant --form-shift 1.05
```

## 二、延迟与音质（都是实测值，48 kHz）

| 块大小 | 算法延迟（无共振峰） | 算法延迟（开共振峰） | 每块耗时 / 预算 | 适用 |
| --- | --- | --- | --- | --- |
| 256 | ≈17.5 ms | ≈34 ms | 0.81 / 5.33 ms | 延迟最低，音质最毛糙 |
| **512** | ≈34.8 ms | ≈68 ms | 1.11 / 10.67 ms | 默认，推荐 |
| 1024 | ≈69.5 ms | ≈138 ms | 1.30 / 21.33 ms | 音质最平滑 |

再加上系统音频缓冲，实际总延迟通常再多 20~40 ms（界面状态栏会给出估算）。
音准：440 Hz 正弦变调 ±12 半音，主频误差 ≤0.2%；半音数为 0 时是**严格原声直通**（不引入任何变化）。
共振峰校正（v0.2 起新增）：男声 +6 半音后，谱质心从 1625 Hz 拉回到 1207 Hz（接近原 880 Hz），
听感上「音高升高但仍是男声」。

变调部分用的是颗粒重叠相加，声音会带一点「金属感/电子感」，这是这类算法（包括大多数开源变声器）的固有特点，
不是配置问题；块大小取 1024 会更平滑，代价是延迟更大。

### v0.4 新增：15 种效果器可任意组合（总开销 < 4ms/块）

v0.4 起内置一个 15 效果器 rack（参考 sioaeko/OpenVoiceChanger 设计），可以同时启用多个、独立调强度：

```
降噪门 / 机器人(环形调制) / 气音 / 电话音 / 失真 / 位破坏 / 合唱 / 回声 /
混响 / 音色 EQ（3 段）/ 压缩器 / 颤音 / 震音 / 呼吸声 / 静音节能
```

GUI 顶部「预设」按钮已预置了 30 个角色，每个都用具体的多效果组合（见下方「预设一览」表）。

### v0.5 新增：微抖动 jitter/shimmer（让声音「更像真人」）

纯音听起来像「机器处理过」，真人说话总有微小的基频/振幅扰动。v0.5 起内置一个 **JitterShimmer** 效果器：

- **jitter（基频抖动）**：基频在 ±0.5% × `jitter_pct` 个采样内做亚采样随机扰动，30Hz 单极 LPF 平滑。真人 jitter 通常 0.5-1.5%。
- **shimmer（振幅抖动）**：振幅乘上 1 + envelope，`shimmer_db` 控制最大变化 dB，50Hz 单极 LPF 平滑。真人 shimmer 通常 0.3-1dB。

用法（API）：

```python
cfg = {
    "jitter_shimmer": {"enabled": True, "jitter_pct": 1.0, "shimmer_db": 0.5},
}
y = pipe.process(x, semitones=0.0, effects_config=cfg)
```

参数建议：
- 自然语音：`jitter_pct=1.0, shimmer_db=0.5`（默认，最佳起点）
- 想要更「真实」：`jitter_pct=1.5, shimmer_db=1.0`
- 想要「机器人感」（已用过强 EQ / 失真）：`jitter_pct=2.0~3.0` 会显得更粗野
- 关闭后严格直通（无任何处理）

性能：单块均值 0.24ms（最大 0.32ms），预算 10.67ms → 富余 33×，可放心叠加到任何预设上。

如果你想完全自定义，可以在代码里这样调用：

```python
import voice_core as vc
import numpy as np

sr, B = 48000, 512
pipe = vc.VoicePipeline(sr, B, formant_correct=True)

# 例：自制「机器人 + 一点点回声 + 低音 +3dB + 微抖动」
cfg = {
    "robot":      {"enabled": True, "hz": 80},
    "echo":       {"enabled": True, "delay_ms": 130, "feedback": 0.35, "mix": 0.45},
    "tone_eq":    {"enabled": True, "bass_db": 3.0},
    "jitter_shimmer": {"enabled": True, "jitter_pct": 1.0, "shimmer_db": 0.5},
}

x = np.random.randn(B).astype(np.float32) * 0.3
y = pipe.process(x, semitones=3.0, form_shift_ratio=1.05,
                 effects_config=cfg)
```

或者离线处理整个文件：

```python
import soundfile as sf, voice_core as vc
x, sr = sf.read("in.wav", framing="float32")
y = vc.process_offline(x, sr, semitones=3.0, form_shift_ratio=1.05,
                       formant_correct=True, effects_config=cfg)
sf.write("out.wav", y, sr)
```

## 三、在游戏 / QQ / 微信里用（虚拟声卡）

你机器上已经装了 **VB-Audio Virtual Cable**（VB-CABLE），所以不用再装东西：

1. 变声器里「输出到」选 **`CABLE Input (VB-Audio Virtual Cable)`**（列表里带 `[MME]` 的那个），点开始变声。
2. 在游戏语音 / QQ / 微信 / Discord 的**麦克风（输入设备）**设置里，选 **`CABLE Output (VB-Audio Virtual Cable)`**。
   这样对方听到的就是变声后的声音。
3. 想自己也听到变声效果：Windows「设置 → 系统 → 声音 → 更多声音设置 → 录音 → CABLE Output → 属性 →
   **侦听** → 勾选『侦听此设备』→ 播放选你的耳机」。⚠️ 记得戴耳机，否则会啸叫。
4. 顺序很重要：**先点开始变声，再进游戏/语音**，不然对方可能收不到声音。
5. 若某个软件里找不到 `CABLE Output`，把它的输入设备刷新一遍或重启该软件。

> 小提示：列表里同一个设备会以 `[MME]`、`[Windows WASAPI]`、`[DirectSound]` 等出现多次，那是不同的驱动通道。
> 双工流要求输入输出属于同一个驱动；混选不同驱动时本程序会自动改成「输入输出分开开流」（状态栏会提示），
> 依然能用，只是理论上会有极小的时钟漂移。

## 四、常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 提示「打不开音频流」 | 麦克风/扬声器被微信、QQ、游戏、OBS、浏览器独占，关掉再试；或换采样率 44100；蓝牙耳机常常只有放音没有麦克风通道 |
| 完全没有声音 | 输出设备选错；或选的是虚拟声卡（那本来就不会有声音，要按第三节在游戏里选 CABLE Output） |
| 声音太小 | Windows「设置 → 系统 → 声音 → 输入 → 音量」调大，或用界面里的「输出音量」滑块（可放大到 2 倍） |
| 状态栏出现 xrun / 声音断续 | 回放卡顿：块大小调大（1024），采样率换 44100，关掉其他吃 CPU 的程序 |
| 音调没有变化 | 半音数确实是 0，往上拖一点 |
| 自己听到自己延迟很大 | 用有线耳机，别用蓝牙；块大小调小到 256 |

声音处理全部在本机完成，不联网、不录音上传。

## 预设一览（v0.4.0 起共 30 项，30 个全部带多效果组合）

按分组显示，每项写明半音数、共振峰位移比、效果组合。**共振峰位移比 >1** = 嘴更小（女声/儿童），**<1** = 嘴更大（男声/大叔）。

v0.4 起每个预设都预设了具体的效果组合，让声音更像真角色，而不是单调的「升降调+单效果」。

| 分组 | 预设 | 半音 | 共振峰 | 效果组合 |
|---|---|---:|---:|---|
| 原声 | 原声 | 0 | 1.00 | — |
| 女性 | 真女声 | +3 | 1.05 | 高频 +1.5dB |
|  | 御姐 | +4 | 1.10 | 0.15 气声 + EQ 低/高频各 +1dB |
|  | 萌妹 | +6 | 1.18 | 0.20 气声 + 高频 +1dB |
|  | 嗲嗲 | +5 | 1.20 | 0.30 气声 + 低-2dB 高+2dB |
|  | 客服女 | +2 | 1.05 | 压缩（-20dB / 3:1） |
|  | 播音女 | +1 | 1.02 | 压缩（-18dB / 4:1）+ EQ |
|  | 少妇 | +2 | 0.96 | 0.2 混响 + 低+1dB |
|  | 老奶奶 | −6 | 0.88 | 4Hz 震音 + 0.3 混响 + 低-2dB |
| 男性 | 真男声 | −3 | 0.95 | 低+1dB |
|  | 大叔 | −5 | 0.90 | 低+2dB + 0.10 气声 |
|  | 恶魔 | −6 | 0.85 | 失真 drive 1.5 + 低+3dB |
|  | 磁性男 | −2 | 0.92 | 低+1dB + 压缩（-18dB / 2.5:1） |
|  | 客服男 | −1 | 0.95 | 压缩（-20dB / 3:1） |
|  | 播音男 | 0 | 0.97 | 压缩（-18dB / 4:1）+ EQ |
|  | 老人 | −7 | 0.85 | 低+3dB + 0.2 混响 + 压缩 |
| 童声 | 萝莉 | +7 | 1.18 | 0.30 气声 |
|  | 正太 | +5 | 1.10 | 0.20 气声 |
|  | 小孩 | +9 | 1.20 | 0.40 气声 + 高频 +1dB |
|  | 娃娃音 | +12 | 1.20 | 合唱 + 0.50 气声 |
|  | 小猫 | +4 | 1.15 | 0.30 气声 |
|  | 花栗鼠 | +7 | 1.20 | 位破坏 0.3 + 0.40 气声 |
|  | 汤姆猫 | +10 | 1.18 | 位破坏 0.2 + 0.40 气声 |
| 特效 | 机器人 | 0 | 1.00 | robot 60Hz |
|  | 电音女王 | +4 | 1.10 | robot 60Hz + 合唱 |
|  | 外星人 | +2 | 1.10 | robot 200Hz + 0.3 混响 |
|  | 兽人 | −5 | 0.85 | 40ms 短回声 + 低+3dB + 失真 |
|  | 幽灵 | −2 | 0.95 | 350ms 长回声 + 0.4 混响 + 合唱 |
|  | 回声 | 0 | 1.00 | 250ms echo |
|  | 电话音 | 0 | 1.00 | telephone 带通 |

可用的效果（`voice_core.EFFECT_NAMES`）：`noise_gate robot whisper telephone distortion bitcrush chorus echo reverb tone_eq compressor output_gain tremolo vibrato breath jitter_shimmer silence_saver`。

**参考来源**（数据来自 GitHub 公开项目；本项目只 port 数值表，不复制源码）：
- [suer781/MaidMic](https://github.com/suer781/MaidMic) — 萝莉/大叔/花栗鼠（Apache-2.0）
- [neboyang/VoiceChanger](https://github.com/neboyang/VoiceChanger) — KITTY/ROSE/WOMAN/UNCLE/MAN/TOM（Apache-2.0）
- [lyrebird-voice-changer/lyrebird](https://github.com/lyrebird-voice-changer/lyrebird) — Darth Vader 等行业基线
- [SUONSUN9527/windows-voice-changer](https://github.com/SUONSUN9527/windows-voice-changer) — 兽人/幽灵/外星人
- [sioaeko/OpenVoiceChanger](https://github.com/sioaeko/OpenVoiceChanger) — 16 效果器 rack 设计参考（MIT）

## 五、自测

不接声卡也能验证 DSP（81 项检查：恒等直通、音准、时长、音效、无 NaN/削波、共振峰校正、VoicePipeline 跑通、30 个预设、多效果组合、jitter/shimmer 注入无副作用）：

```bat
.venv\Scripts\python.exe test_voice_core.py
```

想了解「为什么 v0.2 要加共振峰校正」和「调研了哪些同类项目」请看 **[EXPERIENCE.md](EXPERIENCE.md)**。

## 六、重新安装依赖（换机器时）

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
