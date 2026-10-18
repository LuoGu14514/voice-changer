# -*- coding: utf-8 -*-
"""IndexTTS-2.5 本地 HTTP 服务。

把引擎包成 HTTP 接口，让别的进程（比如 QQ 机器人，跑在另一个 Python 版本
的 venv 里）可以直接调用，不必共享解释器。

只用标准库 http.server，不引入 fastapi/uvicorn 之类的额外依赖。

启动（必须用 3.11 的 venv，且工作目录是 ASCII 路径）::

    cd /d D:\\vc
    D:\\vc\\_indextts_venv\\Scripts\\python.exe indextts_server.py --port 9881

接口::

    GET  /health
        {"ok": true, "loaded": true, "version": "2.5", "sr": 22050,
         "load_s": 33.4, "voices": 12, "threads": 16}

    GET  /voices
        {"voices": [{"id": "三月七/开心", "character": "三月七",
                     "emotion": "开心", "lang": "中文",
                     "ref_audio": "D:\\\\...\\\\x.wav"}]}

    POST /tts
        {"text": "...", "ref_audio": "...", "lang": "ZH",
         "duration_factor": 1.0, "diffusion_steps": 20, "num_beams": 1,
         "peak_limit": 0.95}
        也可用 {"voice": "三月七/开心"} 代替 ref_audio。
        成功 -> 200 + audio/wav 字节，附带 X-Audio-* 头
        失败 -> 4xx/5xx + {"error": {"code": "...", "message": "..."}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from indextts_engine import (
    DEFAULT_PEAK_LIMIT,
    HERE,
    IndexTTSEngine,
    describe_exception,
    log_exception,
    normalize_lang,
)

VERSION = "2.5"
EMOTION_PATTERN = re.compile(r"^【(.+?)】(.*)$")
REF_ROOT = os.path.join(HERE, "models", "gsv", "v2ProPlus")

# 单实例模型不能并行推理，所有合成请求串行化。
_INFER_LOCK = threading.Lock()


# ── 音色索引 ────────────────────────────────────────────────────────────
def _humansize(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def scan_voices(root: str = REF_ROOT) -> list[dict]:
    """扫描参考音频目录，产出音色清单。

    目录结构：``<root>/<角色>/reference_audios/<语言>/emotions/<文件名>.wav``，
    文件名形如 ``【开心】你看你看，小家伙有反应了.wav``，方括号里是情绪名，
    后面是参考文本。
    """
    voices: list[dict] = []
    if not os.path.isdir(root):
        return voices
    for character in sorted(os.listdir(root)):
        character_dir = os.path.join(root, character)
        if not os.path.isdir(character_dir):
            continue
        ref_root = os.path.join(character_dir, "reference_audios")
        if not os.path.isdir(ref_root):
            continue
        for lang in sorted(os.listdir(ref_root)):
            emotions_dir = os.path.join(ref_root, lang, "emotions")
            if not os.path.isdir(emotions_dir):
                continue
            for filename in sorted(os.listdir(emotions_dir)):
                if not filename.lower().endswith(".wav"):
                    continue
                stem = os.path.splitext(filename)[0]
                matched = EMOTION_PATTERN.match(stem)
                emotion = matched.group(1).strip() if matched else ""
                prompt_text = (matched.group(2) if matched else stem).strip()
                full = os.path.join(emotions_dir, filename)
                voices.append(
                    {
                        "id": f"{character}/{emotion}" if emotion else character,
                        "character": character,
                        "emotion": emotion,
                        "lang": lang,
                        "ref_audio": full,
                        "prompt_text": prompt_text,
                        "size": _humansize(os.path.getsize(full)),
                    }
                )
    return voices


class VoiceIndex:
    """线程安全的音色查询（按 id / 角色 / 角色+情绪）。"""

    def __init__(self, root: str = REF_ROOT) -> None:
        self._lock = threading.Lock()
        self._root = root
        self._voices: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._by_character: dict[str, list[dict]] = {}
        self.reload()

    def reload(self) -> int:
        voices = scan_voices(self._root)
        by_id: dict[str, dict] = {}
        by_character: dict[str, list[dict]] = {}
        for voice in voices:
            by_id[voice["id"]] = voice
            by_character.setdefault(voice["character"], []).append(voice)
        with self._lock:
            self._voices = voices
            self._by_id = by_id
            self._by_character = by_character
        return len(voices)

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._voices)

    def resolve(self, *, voice: str = "", ref_audio: str = "") -> str:
        """把 voice id / 角色名 / 情绪 解析成参考音频绝对路径。

        给了 ref_audio 就直接用（调用方显式指定优先）。
        """
        if ref_audio and ref_audio.strip():
            return ref_audio.strip()

        key = (voice or "").strip()
        if not key:
            raise KeyError("未指定 voice 或 ref_audio")

        with self._lock:
            exact = self._by_id.get(key)
            if exact is not None:
                return exact["ref_audio"]

            # 支持「角色:情绪」「角色/情绪」「角色 情绪」几种写法
            for separator in (":", "/", " ", "_", "-"):
                if separator in key:
                    character, _, emotion = key.partition(separator)
                    character, emotion = character.strip(), emotion.strip()
                    for candidate in self._by_character.get(character, []):
                        if candidate["emotion"] == emotion:
                            return candidate["ref_audio"]

            candidates = self._by_character.get(key)
            if candidates:
                # 优先中立，否则取第一个
                for candidate in candidates:
                    if candidate["emotion"] in ("中立", "中性", "normal", "neutral"):
                        return candidate["ref_audio"]
                return candidates[0]["ref_audio"]

            # 再退化：只给情绪名，全局搜
            for candidate in self._voices:
                if candidate["emotion"] == key:
                    return candidate["ref_audio"]

            known = ", ".join(sorted(self._by_id)[:12])
            raise KeyError(f"未知音色 {key!r}；可用示例: {known}")

    def count(self) -> int:
        with self._lock:
            return len(self._voices)


# ── HTTP 处理 ───────────────────────────────────────────────────────────
class SpeechHandler(BaseHTTPRequestHandler):
    server_version = f"IndexTTS/{VERSION}"
    protocol_version = "HTTP/1.1"

    # 由 serve() 注入
    engine: IndexTTSEngine
    voices: VoiceIndex
    ready_event: threading.Event
    load_error: str = ""

    # -- 工具 -----------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - 覆盖基类
        sys.stderr.write(
            "[server] %s %s\n" % (self.address_string(), fmt % args)
        )

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("请求体为空")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    # -- 路由 -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - 基类约定的方法名
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path in ("/", "/health", "/api/health"):
                self._send_json(200, self._health_payload())
                return
            if path in ("/voices", "/api/voices"):
                self._send_json(200, {"voices": self.voices.all()})
                return
            self._send_error_json(404, "not_found", f"未知路径 {path}")
        except Exception as exc:  # noqa: BLE001 - 兜底，保证有响应
            self._send_error_json(500, "internal_error", describe_exception(exc))

    def do_POST(self) -> None:  # noqa: N802 - 基类约定的方法名
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path not in ("/tts", "/api/tts", "/synthesize"):
            self._send_error_json(404, "not_found", f"未知路径 {path}")
            return
        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(400, "bad_request", describe_exception(exc))
            return

        if not self.ready_event.is_set():
            detail = self.load_error or "模型仍在加载"
            self._send_error_json(503, "model_not_ready", detail)
            return

        text = str(payload.get("text") or "").strip()
        if not text:
            self._send_error_json(400, "empty_text", "text 不能为空")
            return

        try:
            ref_audio = self.voices.resolve(
                voice=str(payload.get("voice") or ""),
                ref_audio=str(payload.get("ref_audio") or ""),
            )
        except KeyError as exc:
            self._send_error_json(404, "voice_not_found", str(exc.args[0]))
            return

        try:
            with _INFER_LOCK:
                result = self.engine.synthesize(
                    text,
                    ref_audio,
                    lang=normalize_lang(payload.get("lang")),
                    duration_factor=float(payload.get("duration_factor") or 1.0),
                    diffusion_steps=int(payload.get("diffusion_steps") or 0) or None,
                    num_beams=int(payload.get("num_beams") or 1),
                    peak_limit=float(payload.get("peak_limit") or DEFAULT_PEAK_LIMIT),
                    with_bytes=True,
                )
        except FileNotFoundError as exc:
            self._send_error_json(404, "ref_audio_not_found", describe_exception(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 推理失败统一 500
            self._send_error_json(500, "synthesis_failed", log_exception(exc))
            return

        wav = result.pop("wav_bytes")
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("X-Audio-Duration-Ms", str(int(result["dur"] * 1000)))
        self.send_header("X-Audio-Sample-Rate", str(result["sr"]))
        self.send_header("X-Audio-Peak", str(result["peak"]))
        self.send_header("X-Infer-Ms", str(int(result["infer_s"] * 1000)))
        self.send_header("X-Rtf", str(result["rtf"] or 0))
        self.send_header("X-Diffusion-Steps", str(result["diffusion_steps"]))
        self.end_headers()
        self.wfile.write(wav)
        sys.stderr.write(
            "[server] tts ok dur=%.2fs infer=%.2fs rtf=%s peak=%.3f\n"
            % (result["dur"], result["infer_s"], result["rtf"], result["peak"])
        )

    def _health_payload(self) -> dict:
        return {
            "ok": True,
            "loaded": self.ready_event.is_set(),
            "version": VERSION,
            "sr": self.engine.sample_rate,
            "load_s": self.engine.load_seconds,
            "voices": self.voices.count(),
            "threads": self.engine.threads,
            "error": self.load_error,
        }


def _load_engine(
    engine: IndexTTSEngine,
    ready_event: threading.Event,
    holder: dict,
) -> None:
    try:
        load_s = engine.load()
        ready_event.set()
        sys.stderr.write(f"[server] 模型加载完成 {load_s}s\n")
    except Exception as exc:  # noqa: BLE001 - 后台线程必须吞掉异常
        holder["error"] = describe_exception(exc)
        log_exception(exc)
        sys.stderr.write("[server] 模型加载失败\n")


def serve(
    host: str = "127.0.0.1",
    port: int = 9881,
    *,
    threads: int | None = None,
    use_qwen_emo: bool = False,
    preload: bool = True,
) -> None:
    engine = IndexTTSEngine(threads=threads, use_qwen_emo=use_qwen_emo)
    voices = VoiceIndex()
    ready = threading.Event()
    holder: dict[str, str] = {}

    SpeechHandler.engine = engine
    SpeechHandler.voices = voices
    SpeechHandler.ready_event = ready
    SpeechHandler.load_error = ""

    httpd = ThreadingHTTPServer((host, port), SpeechHandler)
    httpd.daemon_threads = True

    sys.stderr.write(
        f"[server] IndexTTS-{VERSION} 监听 http://{host}:{port}  "
        f"音色 {voices.count()} 个  threads={engine.threads}\n"
    )

    if preload:
        loader = threading.Thread(
            target=_load_engine, args=(engine, ready, holder), daemon=True
        )
        loader.start()
    else:
        ready.set()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[server] 收到中断，退出\n")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IndexTTS-2.5 本地 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数，0=逻辑核数")
    parser.add_argument(
        "--qwen-emo",
        action="store_true",
        help="启用 Qwen 文本情感推断（多占约 1.2 GB 内存）",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="启动时不预加载模型（/tts 会返回 503）",
    )
    args = parser.parse_args(argv)

    serve(
        args.host,
        args.port,
        threads=args.threads or None,
        use_qwen_emo=args.qwen_emo,
        preload=not args.no_preload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
