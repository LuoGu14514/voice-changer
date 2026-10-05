# -*- coding: utf-8 -*-
"""简单变声器 —— 实时麦克风变声小工具（tkinter 图形界面）。

用法：
    .venv\\Scripts\\pythonw.exe voice_changer.py                       # 打开界面
    .venv\\Scripts\\python.exe  voice_changer.py --file in.wav -o out.wav --semitones 5

界面里选好麦克风 -> 选好输出设备（耳机，或虚拟声卡如 CABLE Input）-> 点"开始"。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import voice_core as vc

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TK_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    tk = None
    ttk = filedialog = messagebox = None
    TK_IMPORT_ERROR = _exc

try:
    import sounddevice as sd
    SD_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    sd = None
    SD_IMPORT_ERROR = _exc

try:
    import soundfile as sf
    SF_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    sf = None
    SF_IMPORT_ERROR = _exc

MAX_RECORD_SECONDS = 600  # 录音上限 10 分钟，防止内存爆掉


# ------------------------------------------------------------------ 辅助
def list_devices():
    """返回 (输入设备列表, 输出设备列表)，元素为 (index, 显示名)。

    显示名后面带上驱动名（MME / Windows WASAPI …）：同一个声卡在每个驱动下都会
    出现一次，不标出来根本分不清。双工流要求输入输出属于同一个驱动，所以知道
    驱动名很有用。
    """
    ins, outs = [], []
    if sd is None:
        return ins, outs
    try:
        devs = sd.query_devices()
        apis = sd.query_hostapis()
    except Exception:
        return ins, outs

    def label(i, d):
        try:
            api = apis[int(d["hostapi"])]["name"]
        except Exception:
            api = "?"
        return f"{i}: {d['name']}  [{api}]"

    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            ins.append((i, label(i, d)))
        if d["max_output_channels"] > 0:
            outs.append((i, label(i, d)))
    return ins, outs


class SplitStream:
    """降级方案：输入、输出各开一条流，中间用一个小缓冲接力。

    麦克风和虚拟声卡常常属于不同驱动（比如麦克风在 MME、CABLE Input 在
    WASAPI），PortAudio 不允许这种组合组成双工流，会报
    `Illegal combination of I/O devices (PaErrorCode -9993)`。这时分别开流就能用。
    代价是两条流时钟独立，会有极小漂移；缓冲积压过多时丢掉最旧的数据，
    所以延迟不会越拖越大。
    """

    def __init__(self, in_idx, out_idx, samplerate, blocksize, chans, process):
        self._process = process
        self._lock = threading.Lock()
        self._fifo = np.zeros(0, dtype=np.float32)
        self._in = sd.InputStream(device=in_idx, samplerate=samplerate, blocksize=blocksize,
                                  dtype="float32", channels=chans[0], latency="low",
                                  callback=self._in_cb)
        self._out = sd.OutputStream(device=out_idx, samplerate=samplerate, blocksize=blocksize,
                                    dtype="float32", channels=chans[1], latency="low",
                                    callback=self._out_cb)

    def _in_cb(self, indata, frames, time_info, status):
        y = self._process(indata, status)
        with self._lock:
            self._fifo = y if self._fifo.size == 0 else np.concatenate((self._fifo, y))
            if self._fifo.size > 8 * frames:
                self._fifo = self._fifo[-4 * frames:]

    def _out_cb(self, outdata, frames, time_info, status):
        with self._lock:
            have = min(frames, int(self._fifo.size))
            if have:
                outdata[:have, 0] = self._fifo[:have]
                self._fifo = self._fifo[have:]
        if have < frames:
            outdata[have:, 0] = 0.0

    def start(self):
        self._in.start()
        try:
            self._out.start()
        except Exception:
            self.stop()
            raise

    def stop(self):
        for s in (self._in, self._out):
            try:
                s.stop()
            except Exception:
                pass

    def close(self):
        for s in (self._in, self._out):
            try:
                s.close()
            except Exception:
                pass


def save_wav(path: str, data: np.ndarray, samplerate: int) -> None:
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if sf is not None:
        sf.write(path, data, samplerate, subtype="PCM_16")
        return
    import wave  # 标准库兜底
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(samplerate))
        w.writeframes(pcm.tobytes())


def load_audio(path: str):
    if sf is None:
        raise RuntimeError("缺少 soundfile 库，无法读取音频文件")
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


# ------------------------------------------------------------------ 界面
class VoiceChangerApp:
    # 实时预设：复用 voice_core.VOICE_PRESETS（semitones, form_shift_ratio, effect）
    # 把 (semitones, form_shift) 解开给滑块。
    PRESETS = [
        (name, semi, fs, eff)
        for (name, semi, fs, eff) in vc.VOICE_PRESETS
    ]

    def __init__(self, root):
        self.root = root
        root.title("简单变声器 · 实时麦克风变声")
        root.geometry("720x680")
        root.minsize(680, 640)

        # --- 音频线程只读这些裸属性（避免跨线程访问 tk 变量）---
        self._semitones = 0.0
        self._form_shift = 1.0
        self._gain = 1.0
        self._effect = "none"
        self._formant_on = True
        self._rec_on = False
        self._rec_blocks = []
        self._rec_samples = 0
        self._level = 0.0
        self._peak = 0.0
        self._xruns = 0
        self._error = None

        self.stream = None
        self.pipeline = None
        self.samplerate = 48000
        self.blocksize = 512
        self.recording = False
        self._devices_in = []
        self._devices_out = []

        self._build_ui()
        self.refresh_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(80, self._tick)

    # -------------------------------------------------------------- 构建界面
    def _build_ui(self):
        wrap = ttk.Frame(self.root)
        wrap.pack(fill="both", expand=True)

        # 设备
        box = ttk.LabelFrame(wrap, text=" 设备 ")
        box.pack(fill="x", padx=10, pady=5)
        ttk.Label(box, text="麦克风（输入）").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.in_var = tk.StringVar()
        self.in_combo = ttk.Combobox(box, textvariable=self.in_var, state="readonly", width=52)
        self.in_combo.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Label(box, text="输出到（耳机 / 虚拟声卡）").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(box, textvariable=self.out_var, state="readonly", width=52)
        self.out_combo.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(box, text="刷新设备", command=self.refresh_devices
                   ).grid(row=0, column=2, rowspan=2, padx=8)
        box.columnconfigure(1, weight=1)

        # 音调
        pitch = ttk.LabelFrame(wrap, text=" 音调 · 共振峰（v0.2 起新增） ")
        pitch.pack(fill="x", padx=10, pady=5)
        ttk.Label(pitch, text="半音").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.semi_var = tk.DoubleVar(value=0.0)
        ttk.Scale(pitch, from_=-12, to=12, variable=self.semi_var, orient="horizontal",
                  command=self.on_semi).grid(row=0, column=1, sticky="we", padx=6, pady=6)
        self.semi_label = ttk.Label(pitch, text="+0.0 半音 (1.00x)", width=18, anchor="w")
        self.semi_label.grid(row=0, column=2, padx=6)

        ttk.Label(pitch, text="共振峰位移").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.form_var = tk.DoubleVar(value=1.0)
        ttk.Scale(pitch, from_=0.70, to=1.30, variable=self.form_var, orient="horizontal",
                  command=self.on_form).grid(row=1, column=1, sticky="we", padx=6, pady=6)
        self.form_label = ttk.Label(pitch, text="1.00x（原声）", width=18, anchor="w")
        self.form_label.grid(row=1, column=2, padx=6)
        pitch.columnconfigure(1, weight=1)

        # 共振峰校正开关：复杂信号（真人）才打开；纯音/合成信号开了反而会糊
        self.formant_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pitch, text="启用共振峰校正（让升/降调后的声音更像真人）",
                        variable=self.formant_var, command=self.on_formant_toggle
                        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        preset = ttk.Frame(pitch)
        preset.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 8))
        for name, semi, fs, eff in self.PRESETS:
            ttk.Button(preset, text=name, width=7,
                       command=lambda s=semi, f=fs, e=eff, n=name:
                       self.apply_preset(n, s, f, e)
                       ).pack(side="left", padx=2)

        # 音效 / 音量 / 参数
        fx = ttk.LabelFrame(wrap, text=" 音效 · 音量 · 音质 ")
        fx.pack(fill="x", padx=10, pady=5)
        ttk.Label(fx, text="音效").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.effect_var = tk.StringVar(value=vc.EFFECT_LABELS["none"])
        self.effect_combo = ttk.Combobox(
            fx, textvariable=self.effect_var, state="readonly", width=10,
            values=[vc.EFFECT_LABELS[e] for e in vc.EFFECTS])
        self.effect_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        self.effect_combo.bind("<<ComboboxSelected>>", self.on_effect)

        ttk.Label(fx, text="输出音量").grid(row=0, column=2, sticky="e", padx=8, pady=6)
        self.gain_var = tk.DoubleVar(value=1.0)
        ttk.Scale(fx, from_=0.0, to=2.0, variable=self.gain_var, orient="horizontal",
                  command=self.on_gain, length=190).grid(row=0, column=3, sticky="we", padx=6, pady=6)
        self.gain_label = ttk.Label(fx, text="100%", width=6)
        self.gain_label.grid(row=0, column=4, padx=6)
        fx.columnconfigure(3, weight=1)

        cfg = ttk.Frame(fx)
        cfg.grid(row=1, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 8))
        ttk.Label(cfg, text="采样率").pack(side="left")
        self.sr_var = tk.IntVar(value=48000)
        ttk.Combobox(cfg, textvariable=self.sr_var, state="readonly", width=7,
                     values=(48000, 44100)).pack(side="left", padx=(4, 16))
        ttk.Label(cfg, text="块大小（小=延迟低，大=更平滑）").pack(side="left")
        self.block_var = tk.IntVar(value=512)
        ttk.Combobox(cfg, textvariable=self.block_var, state="readonly", width=7,
                     values=(256, 512, 1024)).pack(side="left", padx=4)

        # 控制
        ctl = ttk.LabelFrame(wrap, text=" 控制 ")
        ctl.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(ctl, text="▶ 开始变声", command=self.toggle, width=13)
        self.start_btn.grid(row=0, column=0, padx=8, pady=8)
        self.rec_btn = ttk.Button(ctl, text="● 开始录音", command=self.toggle_record, width=13)
        self.rec_btn.grid(row=0, column=1, padx=8, pady=8)
        ttk.Button(ctl, text="保存录音…", command=self.save_record, width=11
                   ).grid(row=0, column=2, padx=8, pady=8)
        ttk.Button(ctl, text="怎么用于游戏/语音？", command=self.show_help
                   ).grid(row=0, column=3, padx=8, pady=8)

        self.meter = tk.Canvas(ctl, width=200, height=16, bg="#202020", highlightthickness=0)
        self.meter.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 10))
        self.meter_bar = self.meter.create_rectangle(0, 0, 0, 16, fill="#39d353", width=0)
        self.level_label = ttk.Label(ctl, text="  -inf dBFS")
        self.level_label.grid(row=1, column=2, columnspan=2, sticky="w", padx=8, pady=(0, 10))

        # 状态与提示
        self.status = tk.StringVar(value="未启动")
        ttk.Label(wrap, textvariable=self.status, foreground="#0a6", wraplength=640,
                  justify="left").pack(fill="x", padx=12, pady=(4, 2))
        ttk.Label(wrap, foreground="#777", wraplength=640, justify="left",
                  text="提示：用扬声器放出来会再次被麦克风收进去（啸叫），请戴耳机。"
                       "只想自己试听：输出选耳机；要让它进游戏/QQ/微信，见右侧按钮。"
                  ).pack(fill="x", padx=12, pady=(0, 8))

    # -------------------------------------------------------------- 设备
    def refresh_devices(self):
        ins, outs = list_devices()
        self._devices_in, self._devices_out = ins, outs
        self.in_combo["values"] = [n for _, n in ins]
        self.out_combo["values"] = [n for _, n in outs]
        names_in = [n for _, n in ins]
        names_out = [n for _, n in outs]
        if ins and self.in_var.get() not in names_in:
            self.in_var.set(self._default_choice(ins, 0))
        if outs and self.out_var.get() not in names_out:
            self.out_var.set(self._default_choice(outs, 1))
        if not ins or not outs:
            self.status.set(f"没有检测到音频设备（输入 {len(ins)} 个，输出 {len(outs)} 个），"
                            "请检查系统是否识别到麦克风/扬声器。")

    def _default_choice(self, devices, which):
        try:
            idx = sd.default.device[which]
            for i, name in devices:
                if i == idx:
                    return name
        except Exception:
            pass
        return devices[0][1]

    @staticmethod
    def _parse_index(text):
        try:
            return int(str(text).split(":", 1)[0])
        except Exception:
            return None

    # -------------------------------------------------------------- 参数
    def on_semi(self, _=None):
        semi = float(self.semi_var.get())
        self._semitones = semi
        ratio = vc.semitones_to_ratio(semi)
        self.semi_label.config(text=f"{semi:+.1f} 半音 ({ratio:.2f}x)")

    def on_form(self, _=None):
        fs = float(self.form_var.get())
        self._form_shift = fs
        if abs(fs - 1.0) < 0.005:
            txt = "1.00x（原声）"
        elif fs > 1.0:
            txt = f"{fs:.2f}x（更亮/偏女）"
        else:
            txt = f"{fs:.2f}x（更沉/偏男）"
        self.form_label.config(text=txt)

    def on_formant_toggle(self):
        self._formant_on = bool(self.formant_var.get())
        # 如果流已开，重建 pipeline 让状态生效
        if self.pipeline is not None:
            self.pipeline = None  # 标记失效，_process_block 会重建
        self.status.set("共振峰校正：" + ("开" if self._formant_on else "关"))

    def on_gain(self, _=None):
        g = float(self.gain_var.get())
        self._gain = g
        self.gain_label.config(text=f"{int(round(g * 100))}%")

    def on_effect(self, _=None):
        label = self.effect_var.get()
        for key, text in vc.EFFECT_LABELS.items():
            if text == label:
                self._effect = key
                break

    def apply_preset(self, name, semi, form_shift, effect):
        self.semi_var.set(semi)
        self.on_semi()
        self.form_var.set(form_shift)
        self.on_form()
        self.effect_var.set(vc.EFFECT_LABELS[effect])
        self.on_effect()
        self.status.set(f"已套用预设：{name}（{semi:+.0f} 半音, 共振峰 ×{form_shift:.2f}）")

    # -------------------------------------------------------------- 启停
    def toggle(self):
        if self.stream is not None:
            self.stop()
        else:
            self.start()

    def start(self):
        if sd is None:
            messagebox.showerror("缺少依赖", f"sounddevice 未安装或加载失败：{SD_IMPORT_ERROR}")
            return
        in_idx = self._parse_index(self.in_var.get())
        out_idx = self._parse_index(self.out_var.get())
        if in_idx is None or out_idx is None:
            messagebox.showwarning("请先选择设备", "请在列表里选择麦克风和输出设备。")
            return

        self.blocksize = int(self.block_var.get())
        wanted_sr = int(self.sr_var.get())
        candidates_sr = [wanted_sr] + [s for s in (48000, 44100) if s != wanted_sr]
        try:
            default_sr = int(float(sd.query_devices(out_idx)["default_samplerate"]))
            if default_sr not in candidates_sr:
                candidates_sr.append(default_sr)
        except Exception:
            pass

        last_err = None
        note = ""
        for sr in candidates_sr:
            for chans in ((1, 1), (2, 2)):
                pipeline = vc.VoicePipeline(sr, self.blocksize,
                                            formant_correct=self._formant_on,
                                            align=True)
                pipeline.set_effect(self._effect)
                # 先把状态挂上再开流：回调可能在 start() 里就跑起来
                self.pipeline = pipeline
                stream = None
                for maker in ("duplex", "split"):
                    try:
                        if maker == "duplex":
                            stream = sd.Stream(device=(in_idx, out_idx), samplerate=sr,
                                               blocksize=self.blocksize, dtype="float32",
                                               channels=chans, latency="low",
                                               callback=self._callback)
                        else:
                            stream = SplitStream(in_idx, out_idx, sr, self.blocksize, chans,
                                                 self._process_block)
                        stream.start()
                    except Exception as exc:
                        last_err = exc
                        if stream is not None:
                            try:
                                stream.close()
                            except Exception:
                                pass
                            stream = None
                        continue
                    note = "" if maker == "duplex" else " ｜ 输入/输出分开开流"
                    break
                if stream is None:
                    continue
                self.stream = stream
                self.samplerate = sr
                self._error = None
                self._xruns = 0
                self._rec_blocks = []
                self._rec_samples = 0
                latency_ms = (pipeline.latency_samples + 2 * self.blocksize) / sr * 1000.0
                self.start_btn.config(text="■ 停止")
                self.status.set(f"运行中：{sr} Hz ｜ 块 {self.blocksize} ｜ 声道 {chans} ｜ "
                                f"算法延迟 ≈ {latency_ms:.0f} ms"
                                f"（叠加系统缓冲通常再多 20~40 ms）{note}")
                return
        self.pipeline = self.stream = None
        messagebox.showerror(
            "打不开音频流",
            f"错误：{last_err}\n\n常见原因：\n"
            "· 麦克风/扬声器被别的程序独占（微信、QQ、游戏、OBS、浏览器）\n"
            "· 采样率设备不支持，换 44100 或 48000 再试\n"
            "· 蓝牙耳机只有 A2DP（放音），没有麦克风通道，换有线耳机或内置麦克风\n"
            "· 输出设备选错，比如选了一个只有输入通道的设备\n"
            "· 输入和输出属于不同驱动（列表末尾 [方括号] 里的名字）时已自动改成分开开流；\n"
            "  若仍打不开，尽量选同一个驱动的两个设备，比如都是 [MME]")

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        self._rec_on = False
        if self.recording:
            self.recording = False
            self.rec_btn.config(text="● 开始录音")
        self.start_btn.config(text="▶ 开始变声")
        self.status.set("已停止")

    def _callback(self, indata, outdata, frames, time_info, status):
        """双工流的回调：算完直接把结果写进 outdata。"""
        y = self._process_block(indata, status)
        try:
            if outdata.ndim > 1:
                outdata[:, :] = y[:, None]
            else:
                outdata[:] = y
        except Exception as exc:  # 出错就静音，别让回调把整个流带崩
            self._error = repr(exc)
            outdata.fill(0.0)

    def _process_block(self, indata, status=None):
        """把一块输入变成要播的一块输出（在音频线程里跑，不能碰界面）。"""
        try:
            if status:
                self._xruns += 1
            x = indata.mean(axis=1) if indata.ndim > 1 else indata
            # 共振峰开关切换会让现有 pipeline 失效：就地重建一次
            if self.pipeline is None and self.stream is not None:
                self.pipeline = vc.VoicePipeline(self.samplerate, self.blocksize,
                                                 formant_correct=self._formant_on,
                                                 align=True)
                self.pipeline.set_effect(self._effect)
            y = self.pipeline.process(x, semitones=self._semitones,
                                       form_shift_ratio=self._form_shift,
                                       effect=self._effect)
            if self._gain != 1.0:
                y = y * self._gain
            y = np.clip(y, -1.0, 1.0)
            self._level = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
            self._peak = float(np.max(np.abs(y))) if y.size else 0.0
            if self._rec_on:
                self._rec_blocks.append(y.copy())
                self._rec_samples += y.size
                if self._rec_samples > MAX_RECORD_SECONDS * self.samplerate:
                    self._rec_on = False
            return y
        except Exception as exc:  # 出错就静音，别让回调把整个流带崩
            self._error = repr(exc)
            return np.zeros(int(indata.shape[0]), dtype=np.float32)

    # -------------------------------------------------------------- 录音
    def toggle_record(self):
        if self.stream is None:
            messagebox.showinfo("先开始变声", "请先点「开始变声」，再录音。")
            return
        self.recording = not self.recording
        self._rec_on = self.recording
        if self.recording:
            self._rec_blocks = []
            self._rec_samples = 0
            self.rec_btn.config(text="● 停止录音")
        else:
            self.rec_btn.config(text="● 开始录音")
            self.status.set(f"录音已停止，共 {self.recorded_seconds():.1f} 秒，"
                            "点「保存录音…」导出 wav")

    def recorded_seconds(self):
        return self._rec_samples / float(self.samplerate or 48000)

    def save_record(self):
        if self.recording:
            self.toggle_record()
        blocks = list(self._rec_blocks)
        if not blocks:
            messagebox.showinfo("没有录音", "还没有录到内容：先「开始变声」→「开始录音」。")
            return
        default = "变声_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        path = filedialog.asksaveasfilename(
            title="保存录音", defaultextension=".wav", initialfile=default,
            filetypes=[("WAV 音频", "*.wav")], initialdir=HERE)
        if not path:
            return
        data = np.concatenate(blocks)
        try:
            save_wav(path, data, self.samplerate)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status.set(f"已保存：{path}（{data.size / max(1, self.samplerate):.1f} 秒）")

    # -------------------------------------------------------------- 帮助
    def show_help(self):
        messagebox.showinfo(
            "怎么让它出现在游戏 / 微信 / QQ 里",
            "要让变声后的声音被别的软件当成麦克风，需要一个虚拟声卡：\n\n"
            "1. 装 VB-Audio Virtual Cable（免费，官网 vb-audio.com/Cable 下载，装完重启）\n"
            "2. 本工具「输出到」选 CABLE Input (VB-Audio Virtual Cable)\n"
            "3. 微信/QQ/游戏/直播软件的『麦克风』选 CABLE Output (VB-Audio Virtual Cable)\n"
            "4. 想自己同时听到：Windows 声音设置里把 CABLE Output 设为『侦听此设备』并选耳机\n\n"
            "注意：真实麦克风一次只能被一个程序占用；走虚拟声卡时，其他软件的麦克风要选\n"
            "CABLE Output，不要再选真实麦克风。\n"
            "只是自己试听效果的话，把「输出到」直接选耳机就行。")

    # -------------------------------------------------------------- 定时刷新
    def _tick(self):
        db = 20.0 * np.log10(self._level + 1e-9)
        width = int(max(0.0, min(1.0, (db + 60.0) / 60.0)) * 200)
        color = "#f85149" if self._peak >= 0.99 else "#39d353"
        self.meter.coords(self.meter_bar, 0, 0, width, 16)
        self.meter.itemconfig(self.meter_bar, fill=color)
        self.level_label.config(
            text=f"{db:6.1f} dBFS" + ("   削波!" if self._peak >= 0.99 else ""))

        if self._error is not None and self.stream is not None:
            err = self._error
            self._error = None
            self.stop()
            self.status.set("⚠ 音频线程出错，已停止：" + err)
        elif self.stream is not None:
            base = self.status.get().split("  ｜ 缓冲")[0].split("  ｜ 录音")[0]
            if self._xruns:
                self.status.set(base + f"  ｜ 缓冲异常 {self._xruns} 次（试试更大的块大小）")
            elif self._rec_on:
                self.status.set(base + f"  ｜ 录音中 {self.recorded_seconds():.1f}s")
            elif self.recording:
                self.status.set(base + "  ｜ 录音已到 10 分钟上限，自动停止")
                self.recording = False
                self.rec_btn.config(text="● 开始录音")
        self.root.after(80, self._tick)

    def on_close(self):
        self._rec_on = False
        self.stop()
        self.root.destroy()


# ------------------------------------------------------------------ 命令行
def run_cli(argv):
    parser = argparse.ArgumentParser(description="文件变声（离线处理）")
    parser.add_argument("--file", required=True, help="输入音频（wav/flac/ogg 等）")
    parser.add_argument("-o", "--out", default=None, help="输出 wav，默认 xxx_changed.wav")
    parser.add_argument("--semitones", type=float, default=5.0, help="半音数，正=升高，负=降低")
    parser.add_argument("--effect", default="none", choices=list(vc.EFFECTS))
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--formant", action="store_true",
                        help="启用共振峰校正（让声音更像真人；纯音测试信号不要加）")
    parser.add_argument("--form-shift", type=float, default=1.0,
                        help="共振峰位移倍数（1.0=拉回原位, >1=偏女, <1=偏男）")
    args = parser.parse_args(argv)

    x, sr = load_audio(args.file)
    y = vc.process_offline(x, sr, semitones=args.semitones, effect=args.effect,
                           block_size=args.block, gain=args.gain,
                           formant_correct=args.formant,
                           form_shift_ratio=args.form_shift)
    out = args.out or os.path.splitext(args.file)[0] + "_changed.wav"
    save_wav(out, y, sr)
    print(f"完成：{out}  （{sr} Hz, {y.size / sr:.2f}s, {args.semitones:+.1f} 半音, "
          f"音效={args.effect}, 共振峰 ×{args.form_shift:.2f}）")
    return 0


def main():
    argv = sys.argv[1:]
    if argv:
        if sf is None:
            print(f"错误：需要 soundfile 才能处理音频文件：{SF_IMPORT_ERROR}", file=sys.stderr)
            return 2
        return run_cli(argv)
    if sd is None:
        print(f"错误：sounddevice 未安装/加载失败：{SD_IMPORT_ERROR}", file=sys.stderr)
        print("请运行： .venv\\Scripts\\pip install numpy sounddevice soundfile", file=sys.stderr)
        return 2
    if tk is None:
        print(f"错误：tkinter 不可用：{TK_IMPORT_ERROR}", file=sys.stderr)
        return 2
    root = tk.Tk()
    VoiceChangerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
