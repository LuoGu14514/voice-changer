# -*- coding: utf-8 -*-
"""TTS 文字转语音工具 —— IndexTTS-2.5，角色音色克隆。

输入文字，用指定角色的参考音频做零样本克隆合成。

用法：
    .venv\\Scripts\\pythonw.exe tts_app.py

依赖：
    - 主 venv：tkinter / sounddevice / soundfile（GUI 与播放）
    - _indextts_venv：Python 3.11 + IndexTTS-2.5（实际合成，常驻 worker）

架构：
    GUI 进程（3.12） --JSON-Lines over stdin/stdout-->  worker 进程（3.11）
    模型只在 worker 里加载一次（36 秒），之后每次合成复用。
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
import sounddevice as sd
import soundfile as sf

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


class TTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TTS 语音合成 · IndexTTS-2.5")
        self.root.geometry("880x680")
        self.root.minsize(760, 560)

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

        self._build_ui()
        self._load_characters()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._drain)
        # 启动时就把模型拉起来，别等第一次点合成
        self.root.after(300, self.start_worker)

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        top = ttk.LabelFrame(self.root, text="角色")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="角色：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.cb_char = ttk.Combobox(
            top, values=CHARACTERS, state="readonly", width=14
        )
        self.cb_char.grid(row=0, column=1, sticky="w")
        self.cb_char.bind("<<ComboboxSelected>>", lambda e: self._load_characters())

        ttk.Label(top, text="情绪参考：").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.cb_emo = ttk.Combobox(top, state="readonly", width=12)
        self.cb_emo.grid(row=0, column=3, sticky="w")
        self.cb_emo.bind("<<ComboboxSelected>>", lambda e: self._show_ref_text())

        self.lbl_ref = ttk.Label(top, text="", foreground="#555", wraplength=820)
        self.lbl_ref.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        mid = ttk.LabelFrame(self.root, text="文本")
        mid.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(mid, wrap="word", height=9, font=("Microsoft YaHei UI", 11))
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)
        self.txt.insert("1.0", "你好呀，我是三月七。今天的星穹列车，也要一起出发吗？")

        self.lbl_count = ttk.Label(mid, text="0 字", foreground="#666")
        self.lbl_count.pack(anchor="e", padx=10, pady=(0, 4))
        self.txt.bind("<KeyRelease>", lambda e: self._update_count())
        self._update_count()

        prm = ttk.LabelFrame(self.root, text="参数")
        prm.pack(fill="x", **pad)

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
        self.log = tk.Text(logf, height=7, wrap="word", state="disabled",
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

    # ---------------- worker ----------------
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
                text=f"完成 {msg.get('dur')}s / RTF {msg.get('rtf')}", foreground="#0a6"
            )
            self._log(
                f"合成完成 -> {os.path.basename(self.last_output)} "
                f"{msg.get('dur')}s peak={msg.get('peak')} "
                f"推理 {msg.get('infer_s')}s RTF={msg.get('rtf')}"
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
        if not self.worker_ready:
            if self.worker_loading:
                messagebox.showinfo("请稍等", "模型还在加载中，几秒后再试。")
            else:
                messagebox.showwarning("未就绪", "引擎未启动，请稍等或重启程序。")
            return

        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("没有文本", "请先输入要合成的文字。")
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
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        sd.stop()
        sd.play(data, sr)
        self.btn_stop.configure(state="normal")
        self._log(f"播放 {os.path.basename(path)} ({len(data)/sr:.2f}s)")

    def on_play(self):
        if self.last_output and os.path.isfile(self.last_output):
            try:
                self._play_path(self.last_output)
            except Exception as e:
                messagebox.showerror("播放失败", str(e))

    def on_stop_play(self):
        sd.stop()
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
    TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
