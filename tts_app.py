# -*- coding: utf-8 -*-
"""TTS 文字转语音工具 —— 双后端：IndexTTS-2.5（本地克隆）/ MiniMax（云端）。

输入文字，用指定角色的音色合成语音。

用法：
    python tts_app.py

依赖：
    - GUI 进程：tkinter / numpy / sounddevice / soundfile（播放）
      系统 Python 3.12 即可：
          python -m pip install soundfile sounddevice
    - IndexTTS 后端额外需要 _indextts_venv（Python 3.11 + IndexTTS-2.5）
    - MiniMax 后端只需要标准库（minimax_tts.py 零依赖）

架构：
    GUI 进程（3.12） --JSON-Lines over stdin/stdout-->  worker 进程（3.11）
    模型只在 worker 里加载一次（30-70 秒），之后每次合成复用。

注意：
    IndexTTS worker 稳态占用 ~7.5 GB 内存，本机内存吃紧时会在加载阶段崩。
    所以默认后端是 MiniMax（云端，不吃本地内存），IndexTTS 改成选中后才懒加载。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import minimax_tts
from minimax_tts import MiniMaxError, MiniMaxTTS

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - 缺 PortAudio 时降级
    sd = None
try:
    import soundfile as sf
except Exception:  # pragma: no cover - 缺 libsndfile 时降级
    sf = None

HERE = os.path.dirname(os.path.abspath(__file__))

# kaldifst 的 C++ 代码打不开含中文的路径，必须用 ASCII 目录。
# 项目里建了 junction: D:\vc -> D:\代码\voice_changer
ASCII_ROOT = r"D:\vc"
if HERE.isascii() and os.path.isdir(HERE):
    WORK_ROOT = HERE
elif os.path.isdir(ASCII_ROOT):
    WORK_ROOT = ASCII_ROOT
else:
    WORK_ROOT = HERE

INDEXTTS_PYTHON = os.path.join(WORK_ROOT, "_indextts_venv", "Scripts", "python.exe")
WORKER_SCRIPT = os.path.join(WORK_ROOT, "indextts_worker.py")
MODELS_ROOT = os.path.join(WORK_ROOT, "models", "gsv", "v2ProPlus")
OUTPUT_DIR = os.path.join(WORK_ROOT, "tts_outputs")

CHARACTERS = ["三月七", "青雀"]

EMOTION_RE = re.compile(r"^【(.+?)】(.+)\.wav$")

BACKEND_INDextts = "indextts"
BACKEND_MINIMAX = "minimax"


def list_emotions(character: str):
    """返回 [(情绪名, 完整路径, 参考文本)]，按情绪名排序。"""
    folder = os.path.join(MODELS_ROOT, character, "reference_audios", "中文", "emotions")
    if not os.path.isdir(folder):
        return []
    items = []
    for fn in os.listdir(folder):
        m = EMOTION_RE.match(fn)
        if not m:
            continue
        emo, text = m.group(1), m.group(2)
        items.append((emo, os.path.join(folder, fn), text))
    items.sort(key=lambda x: x[0])
    return items


def grouped_preset_names():
    """返回 [(分组, [预设名, ...]), ...]，用于 MiniMax 角色下拉。"""
    return [(g, [p["name"] for p in items]) for g, items in minimax_tts.grouped_presets()]


class TTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TTS 语音合成 · IndexTTS-2.5 / MiniMax")
        self.root.geometry("940x740")
        self.root.minsize(820, 620)

        self.backend_var = tk.StringVar(value=BACKEND_MINIMAX)

        self.worker = None            # subprocess.Popen
        self.worker_ready = False
        self.worker_loading = False
        self.reader_thread = None
        self.msgs = queue.Queue()     # reader -> GUI
        self.req_id = 0
        self.pending_id = None
        self.busy = False
        self.last_output = None
        self.emotions = []

        self.mm_client = None
        self.mm_thread = None

        self._build_ui()
        self._load_characters()
        self._load_mm_presets()
        self._load_mm_key()
        self._apply_backend()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._drain)

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ---- 后端选择 ----
        be = ttk.LabelFrame(self.root, text="后端")
        be.pack(fill="x", **pad)
        self.rb_mm = ttk.Radiobutton(
            be, text="MiniMax 云端（快、不占本地内存）",
            value=BACKEND_MINIMAX, variable=self.backend_var,
            command=self._apply_backend,
        )
        self.rb_mm.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.rb_it = ttk.Radiobutton(
            be, text="IndexTTS-2.5 本地克隆（慢、吃 ~7.5GB 内存）",
            value=BACKEND_INDextts, variable=self.backend_var,
            command=self._apply_backend,
        )
        self.rb_it.grid(row=0, column=1, sticky="w", padx=(24, 0))

        # ---- MiniMax 面板 ----
        self.frm_mm = ttk.LabelFrame(self.root, text="MiniMax 参数")
        self.frm_mm.pack(fill="x", **pad)

        ttk.Label(self.frm_mm, text="API Key：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.ent_key = ttk.Entry(self.frm_mm, width=42, show="*")
        self.ent_key.grid(row=0, column=1, sticky="w")
        ttk.Button(self.frm_mm, text="保存", width=6, command=self.on_save_key).grid(
            row=0, column=2, padx=6)
        ttk.Button(self.frm_mm, text="测试连接", width=10, command=self.on_test_key).grid(
            row=0, column=3, padx=2)

        ttk.Label(self.frm_mm, text="模型：").grid(row=1, column=0, sticky="w", padx=6)
        self.cb_model = ttk.Combobox(
            self.frm_mm, values=list(minimax_tts.MODELS), state="readonly", width=22
        )
        self.cb_model.set(minimax_tts.DEFAULT_MODEL)
        self.cb_model.grid(row=1, column=1, sticky="w")

        ttk.Label(self.frm_mm, text="角色：").grid(row=1, column=2, sticky="w", padx=(18, 0))
        self.cb_mm_char = ttk.Combobox(self.frm_mm, state="readonly", width=18)
        self.cb_mm_char.grid(row=1, column=3, sticky="w")
        self.cb_mm_char.bind("<<ComboboxSelected>>", lambda e: self._show_mm_preset())

        ttk.Label(self.frm_mm, text="语速：").grid(row=2, column=0, sticky="w", padx=6, pady=(6, 0))
        self.sp_mm_speed = ttk.Spinbox(
            self.frm_mm, from_=0.5, to=2.0, increment=0.05, width=8, format="%.2f")
        self.sp_mm_speed.set("1.00")
        self.sp_mm_speed.grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.frm_mm, text="音高：").grid(row=2, column=2, sticky="w", padx=(18, 0), pady=(6, 0))
        self.sp_mm_pitch = ttk.Spinbox(
            self.frm_mm, from_=-12, to=12, increment=1, width=6)
        self.sp_mm_pitch.set("0")
        self.sp_mm_pitch.grid(row=2, column=3, sticky="w", pady=(6, 0))

        ttk.Label(self.frm_mm, text="情感：").grid(row=3, column=0, sticky="w", padx=6, pady=(6, 0))
        self.cb_mm_emo = ttk.Combobox(
            self.frm_mm, values=["自动"] + list(minimax_tts.EMOTIONS), state="readonly", width=14)
        self.cb_mm_emo.set("自动")
        self.cb_mm_emo.grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.frm_mm, text="语言：").grid(row=3, column=2, sticky="w", padx=(18, 0), pady=(6, 0))
        self.cb_mm_lang = ttk.Combobox(
            self.frm_mm,
            values=["Chinese", "Chinese,Yue", "English", "Japanese", "auto"],
            state="readonly", width=14)
        self.cb_mm_lang.set("Chinese")
        self.cb_mm_lang.grid(row=3, column=3, sticky="w", pady=(6, 0))

        self.lbl_mm_preset = ttk.Label(self.frm_mm, text="", foreground="#555", wraplength=880)
        self.lbl_mm_preset.grid(row=4, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 6))

        # ---- IndexTTS 面板 ----
        self.frm_it = ttk.LabelFrame(self.root, text="IndexTTS 角色")
        self.frm_it.pack(fill="x", **pad)

        ttk.Label(self.frm_it, text="角色：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.cb_char = ttk.Combobox(
            self.frm_it, values=CHARACTERS, state="readonly", width=14
        )
        self.cb_char.grid(row=0, column=1, sticky="w")
        self.cb_char.bind("<<ComboboxSelected>>", lambda e: self._load_characters())

        ttk.Label(self.frm_it, text="情绪参考：").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.cb_emo = ttk.Combobox(self.frm_it, state="readonly", width=12)
        self.cb_emo.grid(row=0, column=3, sticky="w")
        self.cb_emo.bind("<<ComboboxSelected>>", lambda e: self._show_ref_text())

        self.lbl_ref = ttk.Label(self.frm_it, text="", foreground="#555", wraplength=880)
        self.lbl_ref.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        # ---- 文本 ----
        mid = ttk.LabelFrame(self.root, text="文本")
        mid.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(mid, wrap="word", height=9, font=("Microsoft YaHei UI", 11))
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)
        self.txt.insert("1.0", "你好呀，我是三月七。今天的星穹列车，也要一起出发吗？")

        self.lbl_count = ttk.Label(mid, text="0 字", foreground="#666")
        self.lbl_count.pack(anchor="e", padx=10, pady=(0, 4))
        self.txt.bind("<KeyRelease>", lambda e: self._update_count())
        self._update_count()

        # ---- 参数（IndexTTS 专用） ----
        prm = ttk.LabelFrame(self.root, text="IndexTTS 参数")
        prm.pack(fill="x", **pad)
        self.prm_frame = prm

        ttk.Label(prm, text="语速系数：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.sp_speed = ttk.Spinbox(
            prm, from_=0.5, to=2.0, increment=0.05, width=8, format="%.2f"
        )
        self.sp_speed.set("1.00")
        self.sp_speed.grid(row=0, column=1, sticky="w")

        ttk.Label(prm, text="扩散步数：").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.sp_steps = ttk.Spinbox(prm, from_=8, to=40, increment=1, width=6)
        self.sp_steps.set("20")
        self.sp_steps.grid(row=0, column=3, sticky="w")

        ttk.Label(prm, text="beam：").grid(row=0, column=4, sticky="w", padx=(18, 0))
        self.sp_beams = ttk.Spinbox(prm, from_=1, to=5, increment=1, width=6)
        self.sp_beams.set("1")
        self.sp_beams.grid(row=0, column=5, sticky="w")

        self.lbl_param_hint = ttk.Label(
            prm,
            text="语速 >1 更慢；扩散步数越低越快但音质下降（20 为默认平衡点）；beam=1 最快",
            foreground="#666",
        )
        self.lbl_param_hint.grid(row=1, column=0, columnspan=6, sticky="w", padx=8, pady=(0, 6))

        # ---- 按钮 ----
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", **pad)

        self.btn_synth = ttk.Button(bar, text="合成", command=self.on_synth, width=10)
        self.btn_synth.pack(side="left", padx=4)
        self.btn_play = ttk.Button(
            bar, text="播放", command=self.on_play, width=8, state="disabled"
        )
        self.btn_play.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(
            bar, text="停止播放", command=self.on_stop_play, width=10, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=4)
        self.btn_save = ttk.Button(
            bar, text="另存为…", command=self.on_save_as, width=10, state="disabled"
        )
        self.btn_save.pack(side="left", padx=4)
        self.btn_open = ttk.Button(
            bar, text="打开目录", command=self.on_open_dir, width=10
        )
        self.btn_open.pack(side="left", padx=4)

        self.lbl_status = ttk.Label(bar, text="", foreground="#0a6")
        self.lbl_status.pack(side="right", padx=8)

        logf = ttk.LabelFrame(self.root, text="日志")
        logf.pack(fill="both", **pad)
        self.log = tk.Text(logf, height=6, wrap="word", state="disabled",
                           font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _update_count(self):
        n = len(self.txt.get("1.0", "end").strip())
        self.lbl_count.configure(text=f"{n} 字")

    def _apply_backend(self):
        """切换后端：显示对应面板 + 隐藏另一个。"""
        backend = self.backend_var.get()
        self.backend = backend
        if backend == BACKEND_MINIMAX:
            self.frm_mm.pack(fill="x", padx=8, pady=4)
            self.frm_it.pack_forget()
            self.prm_frame.pack_forget()
        else:
            self.frm_mm.pack_forget()
            self.frm_it.pack(fill="x", padx=8, pady=4)
            self.prm_frame.pack(fill="x", padx=8, pady=4)
        self.lbl_status.configure(text="", foreground="#0a6")
        if backend == BACKEND_INDextts and self.worker is None and not self.worker_loading:
            self._log("已切到 IndexTTS 本地后端，点「合成」时会懒加载模型（30-70 秒）。")

    def _load_characters(self):
        char = self.cb_char.get() or CHARACTERS[0]
        self.cb_char.set(char)
        self.emotions = list_emotions(char)
        names = [e[0] for e in self.emotions]
        self.cb_emo.configure(values=names)
        if names:
            self.cb_emo.set(names[0])
        else:
            self.cb_emo.set("")
        self._show_ref_text()
        self._log(f"角色 {char}：{len(names)} 个情绪参考")

    def _current_ref(self):
        emo = self.cb_emo.get()
        for name, path, text in self.emotions:
            if name == emo:
                return path, text
        return None, None

    def _show_ref_text(self):
        path, text = self._current_ref()
        if path:
            self.lbl_ref.configure(text=f"参考文本：{text}")
        else:
            self.lbl_ref.configure(text="（没有找到参考音频）")

    # ---------------- MiniMax ----------------
    def _load_mm_presets(self):
        items = []
        self._mm_names = []
        for group, names in grouped_preset_names():
            for n in names:
                items.append(f"{group} · {n}")
                self._mm_names.append(n)
        self.cb_mm_char.configure(values=items)
        if items:
            self.cb_mm_char.set(items[0])
        self._show_mm_preset()

    def _show_mm_preset(self):
        label = self.cb_mm_char.get()
        name = label.split("·")[-1].strip() if label else ""
        p = minimax_tts.get_preset(name)
        if not p:
            self.lbl_mm_preset.configure(text="")
            return
        self.lbl_mm_preset.configure(
            text=f"{p['name']}｜音色 {p['voice_id']}"
                 + (f"｜{p['note']}" if p.get("note") else "")
        )

    def _load_mm_key(self):
        key = ""
        try:
            key = minimax_tts._load_key_from_config()
        except Exception:
            pass
        if key:
            self.ent_key.delete(0, "end")
            self.ent_key.insert(0, key)

    def on_save_key(self):
        key = self.ent_key.get().strip()
        if not key:
            messagebox.showwarning("没有 Key", "请先填入 MiniMax API Key。")
            return
        try:
            path = minimax_tts.save_key_to_config(key)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return
        self._log(f"API Key 已保存到 {path}")

    def _ensure_client(self):
        """拿到一个可用的客户端。必须在 GUI 线程调用（会读 entry 控件）。"""
        key = self.ent_key.get().strip() or minimax_tts._load_key_from_config()
        if not key:
            raise MiniMaxError(
                "缺少 API Key。填在上方输入框点「保存」，或设置环境变量 MINIMAX_API_KEY。"
            )
        if self.mm_client is None or self.mm_client.api_key != key:
            self.mm_client = MiniMaxTTS(
                api_key=key, model=minimax_tts.DEFAULT_MODEL
            )
        return self.mm_client

    def on_test_key(self):
        self._log("测试 MiniMax 连接…")
        try:
            client = self._ensure_client()
            voices = client.list_voices()
            counts = {k: len(v) for k, v in voices.items()
                      if isinstance(v, list) and not k.startswith("base_")}
            self._log(f"连接成功，音色：{counts}")
            self.lbl_status.configure(text="MiniMax 连接正常", foreground="#0a6")
        except Exception as e:
            self._log(f"[错误] {e}")
            self.lbl_status.configure(text="连接失败", foreground="#c00")

    def _mm_synth_worker(self, client, text, voice_id, preset, out_path, model, fmt,
                         pitch=0, language_boost="Chinese"):
        """后台线程：调 MiniMax 合成。

        注意：所有 Tk 控件的值都必须在 GUI 线程里读出来再传进来，
        不能在这个线程里碰 widget（Tk 不是线程安全的）。
        """
        try:
            data = client.synth(
                text, voice_id,
                model=model,
                speed=preset.get("speed", 1.0),
                vol=preset.get("vol", 1.0),
                pitch=int(pitch),
                emotion=preset.get("emotion"),
                fmt=fmt,
                sample_rate=32000,
                channel=1,
                language_boost=language_boost,
                voice_modify=preset.get("modify"),
            )
            with open(out_path, "wb") as f:
                f.write(data)
            self.msgs.put({
                "id": self.pending_id,
                "ok": True,
                "output": out_path,
                "dur": self._wav_duration(out_path),
                "peak": 0.0,
                "infer_s": 0.0,
                "rtf": 0.0,
                "backend": "minimax",
            })
        except Exception as e:
            self.msgs.put({
                "id": self.pending_id,
                "ok": False,
                "error": str(e),
                "backend": "minimax",
            })

    @staticmethod
    def _wav_duration(path):
        """不依赖 soundfile 读 wav 时长（读 RIFF 头）。"""
        try:
            import wave
            with wave.open(path, "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return 0.0

    # ---------------- worker（IndexTTS） ----------------
    def start_worker(self):
        if self.worker is not None or self.worker_loading:
            return
        if not os.path.isfile(INDEXTTS_PYTHON):
            self._log(f"[错误] 找不到 python: {INDEXTTS_PYTHON}")
            self.lbl_status.configure(text="环境缺失", foreground="#c00")
            return

        self.worker_loading = True
        self.lbl_status.configure(text="模型加载中…", foreground="#c80")
        self._log(f"启动 worker: {INDEXTTS_PYTHON}")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.worker = subprocess.Popen(
                [INDEXTTS_PYTHON, "-u", WORKER_SCRIPT],
                cwd=WORK_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except Exception as e:
            self.worker_loading = False
            self._log(f"[错误] 启动失败: {e}")
            self.lbl_status.configure(text="启动失败", foreground="#c00")
            return

        self.reader_thread = threading.Thread(
            target=self._reader_loop, args=(self.worker,), daemon=True
        )
        self.reader_thread.start()

    def _reader_loop(self, proc):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.msgs.put(json.loads(line))
            except Exception:
                self.msgs.put({"ok": False, "error": f"bad line: {line[:200]}"})
        self.msgs.put({"event": "eof", "code": proc.poll()})
        # stderr 尾部用于诊断
        try:
            tail = proc.stderr.read() if proc.stderr else ""
            if tail:
                self.msgs.put({"event": "stderr", "text": tail[-3000:]})
        except Exception:
            pass

    def _drain(self):
        try:
            while True:
                msg = self.msgs.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _handle(self, msg):
        if msg.get("event") == "eof":
            self.worker_ready = False
            self.worker_loading = False
            self.busy = False
            self.worker = None
            self.lbl_status.configure(text="worker 已退出", foreground="#c00")
            self._log(f"worker 退出 code={msg.get('code')}")
            self.btn_synth.configure(state="normal")
            return

        if msg.get("event") == "stderr":
            tail = (msg.get("text") or "").strip()
            if tail:
                self._log("--- worker stderr ---")
                for ln in tail.splitlines()[-12:]:
                    self._log(ln)
            return

        if msg.get("ready"):
            self.worker_ready = True
            self.worker_loading = False
            self.lbl_status.configure(text="引擎已就绪", foreground="#0a6")
            self._log(f"模型加载完成 {msg.get('load_s')}s")
            return

        if msg.get("id") != self.pending_id:
            return

        self.busy = False
        self.pending_id = None
        self.btn_synth.configure(state="normal")

        if msg.get("ok"):
            self.last_output = msg.get("output")
            self.lbl_status.configure(
                text=f"完成 {msg.get('dur')}s", foreground="#0a6"
            )
            self._log(
                f"合成完成 -> {os.path.basename(self.last_output)} "
                f"{msg.get('dur')}s"
                + (f" peak={msg.get('peak')}" if msg.get("peak") else "")
                + (f" 推理 {msg.get('infer_s')}s" if msg.get("infer_s") else "")
            )
            self.btn_play.configure(state="normal")
            self.btn_save.configure(state="normal")
        else:
            err = msg.get("error") or "未知错误"
            self.lbl_status.configure(text="合成失败", foreground="#c00")
            self._log(f"[错误] {err}")
            messagebox.showerror("合成失败", err)

    # ---------------- actions ----------------
    def on_synth(self):
        if self.busy:
            return

        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("没有文本", "请先输入要合成的文字。")
            return

        if self.backend == BACKEND_MINIMAX:
            self._on_synth_minimax(text)
        else:
            self._on_synth_indextts(text)

    def _on_synth_minimax(self, text):
        label = self.cb_mm_char.get()
        name = label.split("·")[-1].strip() if label else ""
        preset = minimax_tts.get_preset(name)
        if not preset:
            messagebox.showwarning("没有角色", "请选择一个角色预设。")
            return
        if len(text) >= 10000:
            messagebox.showwarning(
                "文本过长",
                "MiniMax 同步接口上限 10000 字符，当前 %d 字符。" % len(text),
            )
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(OUTPUT_DIR, f"mm_{preset['name']}_{stamp}.wav")

        self.req_id += 1
        self.pending_id = self.req_id
        self.busy = True
        self.btn_synth.configure(state="disabled")
        self.lbl_status.configure(text="MiniMax 合成中…", foreground="#c80")
        self._log(f"MiniMax 合成：{len(text)} 字，角色 {preset['name']}")

        model = self.cb_model.get() or minimax_tts.DEFAULT_MODEL
        pitch = int(self.sp_mm_pitch.get() or 0)
        lang = self.cb_mm_lang.get() or "Chinese"

        # 客户端在 GUI 线程里建好（要读 entry），worker 只管发请求
        try:
            client = self._ensure_client()
        except Exception as e:
            messagebox.showerror("缺少 API Key", str(e))
            return

        self.mm_thread = threading.Thread(
            target=self._mm_synth_worker,
            args=(client, text, preset["voice_id"], preset, out, model, "wav"),
            kwargs={"pitch": pitch, "language_boost": lang},
            daemon=True,
        )
        self.mm_thread.start()

    def _on_synth_indextts(self, text):
        if not self.worker_ready:
            if self.worker_loading:
                messagebox.showinfo("请稍等", "模型还在加载中，几秒后再试。")
            else:
                self.start_worker()
                messagebox.showinfo("正在启动", "本地引擎正在启动（约 30-70 秒），稍后再点合成。")
            return

        ref, _ = self._current_ref()
        if not ref or not os.path.isfile(ref):
            messagebox.showwarning("缺少参考音频", "当前角色没有可用的参考音频。")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(OUTPUT_DIR, f"{self.cb_char.get()}_{stamp}.wav")

        self.req_id += 1
        self.pending_id = self.req_id
        self.busy = True
        self.btn_synth.configure(state="disabled")
        self.lbl_status.configure(text="合成中…(CPU，约 30-60 秒)", foreground="#c80")
        self._log(f"开始合成：{len(text)} 字，参考 {os.path.basename(ref)}")

        req = {
            "id": self.req_id,
            "text": text,
            "ref": ref,
            "lang": "ZH",
            "output": out,
            "duration_factor": float(self.sp_speed.get() or 1.0),
            "diffusion_steps": int(self.sp_steps.get() or 20),
            "num_beams": int(self.sp_beams.get() or 1),
        }
        try:
            self.worker.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.worker.stdin.flush()
        except Exception as e:
            self.busy = False
            self.btn_synth.configure(state="normal")
            self._log(f"[错误] 发送请求失败: {e}")
            self.lbl_status.configure(text="请求失败", foreground="#c00")

    def _play_path(self, path):
        """播放音频。优先 sounddevice，缺失时退回 winsound / 系统播放器。"""
        if sd is not None and sf is not None:
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            sd.stop()
            sd.play(data, sr)
            self.btn_stop.configure(state="normal")
            self._log(f"播放 {os.path.basename(path)} ({len(data)/sr:.2f}s)")
            return

        # 回退：winsound 只吃 WAV；其它格式交给系统默认播放器
        if os.path.splitext(path)[1].lower() == ".wav":
            try:
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.btn_stop.configure(state="normal")
                self._log(f"播放 {os.path.basename(path)}（winsound）")
                return
            except Exception as e:
                self._log(f"[警告] winsound 播放失败: {e}")
        os.startfile(path)  # noqa: S606 (Windows)
        self._log(f"已用系统播放器打开 {os.path.basename(path)}")

    def on_play(self):
        if self.last_output and os.path.isfile(self.last_output):
            try:
                self._play_path(self.last_output)
            except Exception as e:
                messagebox.showerror("播放失败", str(e))

    def on_stop_play(self):
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        else:
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        self.btn_stop.configure(state="disabled")

    def on_save_as(self):
        if not self.last_output or not os.path.isfile(self.last_output):
            return
        default = os.path.basename(self.last_output)
        dst = filedialog.asksaveasfilename(
            title="另存为",
            defaultextension=".wav",
            initialfile=default,
            filetypes=[("WAV 音频", "*.wav"), ("所有文件", "*.*")],
        )
        if not dst:
            return
        try:
            import shutil

            shutil.copyfile(self.last_output, dst)
            self._log(f"已另存为 {dst}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def on_open_dir(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            os.startfile(OUTPUT_DIR)  # noqa: S606 (Windows)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def on_close(self):
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        if self.worker is not None:
            try:
                self.worker.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.worker.stdin.flush()
                self.worker.wait(timeout=5)
            except Exception:
                pass
            try:
                self.worker.kill()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("vista")
    except Exception:
        pass
    app = TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
