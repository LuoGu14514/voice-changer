# -*- coding: utf-8 -*-
"""Real-time RVC voice changer engine.

Architecture:
  [mic @ device_sr] → resample → 16kHz ring_in
                                       ↓
                               inference thread (every hop_ms):
                                 - pull context from ring_in
                                 - HuBERT → F0 → RVC → 48kHz wav
                                 - put output chunk in output ring
  [output ring @ device_sr] → sounddevice output stream → speakers

Designed for CPU-only inference. Latency ≈ context_ms + hop_ms.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger("rvc_realtime")


# ---------------------------------------------------------------------------
# Ring buffer (lock-free single-producer single-consumer)
# ---------------------------------------------------------------------------
class RingBuffer:
    """Simple SPSC ring buffer with numpy backing."""

    def __init__(self, capacity_samples: int, dtype=np.float32):
        self.buf = np.zeros(capacity_samples, dtype=dtype)
        self.cap = capacity_samples
        self.read_pos = 0
        self.write_pos = 0
        # available = (write - read) mod cap
        self._lock = threading.Lock()  # for multi-thread safety (cheap)

    def write(self, data: np.ndarray):
        with self._lock:
            n = len(data)
            if n == 0:
                return
            end = self.write_pos + n
            if end <= self.cap:
                self.buf[self.write_pos:end] = data
            else:
                first = self.cap - self.write_pos
                self.buf[self.write_pos:] = data[:first]
                self.buf[:n - first] = data[first:]
            self.write_pos = (self.write_pos + n) % self.cap

    def available(self) -> int:
        with self._lock:
            return (self.write_pos - self.read_pos) % self.cap

    def read_latest(self, n: int) -> np.ndarray:
        """Read the most recent n samples."""
        with self._lock:
            avail = (self.write_pos - self.read_pos) % self.cap
            if avail < n:
                return np.zeros(n, dtype=self.buf.dtype)
            # compute start position (write_pos - n)
            start = (self.write_pos - n) % self.cap
            if start + n <= self.cap:
                return self.buf[start:start + n].copy()
            else:
                first = self.cap - start
                return np.concatenate([
                    self.buf[start:],
                    self.buf[:n - first],
                ]).copy()

    def read_n(self, n: int) -> Optional[np.ndarray]:
        """Pop n samples from the read end (advance read_pos by n).
        Returns None if not enough samples available."""
        with self._lock:
            avail = (self.write_pos - self.read_pos) % self.cap
            if avail < n:
                return None
            start = self.read_pos
            if start + n <= self.cap:
                out = self.buf[start:start + n].copy()
            else:
                first = self.cap - start
                out = np.concatenate([
                    self.buf[start:],
                    self.buf[:n - first],
                ]).copy()
            self.read_pos = (self.read_pos + n) % self.cap
            return out

    def clear(self):
        with self._lock:
            self.read_pos = 0
            self.write_pos = 0


# ---------------------------------------------------------------------------
# Real-time RVC engine
# ---------------------------------------------------------------------------
class RealtimeRVC:
    """Real-time voice conversion using a pretrained RVC model.

    Args:
        pth_path: path to RVC .pth checkpoint
        hubert_path: path to chinese-hubert-base (or compatible)
        device_sr: sounddevice I/O sample rate
        context_ms: how much audio to feed into each inference (ms)
        hop_ms: how often to run inference (ms) — also output chunk size
        f0_up_key: semitones to shift F0 (default 0; +12 = octave up)
        device: torch device
    """

    def __init__(
        self,
        pth_path: str,
        hubert_path: str,
        device_sr: int = 48000,
        context_ms: int = 600,
        hop_ms: int = 200,
        f0_up_key: float = 0.0,
        device: str = "cpu",
    ):
        self.pth_path = pth_path
        self.hubert_path = hubert_path
        self.device_sr = device_sr
        self.context_ms = context_ms
        self.hop_ms = hop_ms
        self.f0_up_key = f0_up_key
        self.device = device

        # Internal sample rates
        self.feat_sr = 16000  # HuBERT and F0 work at 16kHz
        self.rvc_sr = 48000   # RVC models typically 48kHz output

        # Sizes in samples (at 16kHz)
        self.context_samples_16k = int(self.feat_sr * context_ms / 1000)
        self.hop_samples_16k = int(self.feat_sr * hop_ms / 1000)

        # Output ring at device_sr
        self.out_hop_samples = int(device_sr * hop_ms / 1000)
        # Padding for tail silence after stop
        self.output_ring_capacity = self.out_hop_samples * 8

        self.ring_in = RingBuffer(capacity_samples=self.context_samples_16k * 2)
        self.ring_out = RingBuffer(capacity_samples=self.output_ring_capacity)

        # State
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._ready_event = threading.Event()
        self._model = None
        self._hubert = None
        self._f0_extractor = None
        self._infer_count = 0
        self._infer_time_ms = 0.0

    # ----------------------------------------------------------------
    # Lazy model loading (called by inference thread)
    # ----------------------------------------------------------------
    def _load_models(self):
        from rvc_infer import HuBERTExtractor, RVCModel
        log.info("Loading RVC models...")
        self._hubert = HuBERTExtractor(self.hubert_path, device=self.device)
        self._model = RVCModel(self.pth_path, device=self.device)
        log.info(f"  Model SR: {self._model.sr_str} → {self._model.sr}Hz")
        self._ready_event.set()

    # ----------------------------------------------------------------
    # Input: called from audio thread (mic callback)
    # ----------------------------------------------------------------
    def push_audio(self, block: np.ndarray, block_sr: int):
        """Push a block of audio from the input stream.

        Args:
            block: mono float32 audio at block_sr
            block_sr: the actual sample rate of the block
        """
        if len(block) == 0:
            return
        # Resample to 16kHz
        if block_sr == self.feat_sr:
            audio_16k = block
        else:
            audio_16k = self._resample(block, block_sr, self.feat_sr)
        self.ring_in.write(audio_16k.astype(np.float32))

    # ----------------------------------------------------------------
    # Output: called from audio thread (speaker callback)
    # ----------------------------------------------------------------
    def pull_audio(self, n_samples: int) -> np.ndarray:
        """Pull n_samples of output audio at device_sr.

        Returns zeros if not enough ready."""
        out = self.ring_out.read_n(n_samples)
        if out is None:
            # pad with zeros for underrun
            out = np.zeros(n_samples, dtype=np.float32)
        return out

    # ----------------------------------------------------------------
    # Inference thread
    # ----------------------------------------------------------------
    def _inference_loop(self):
        # Load models on first run
        self._load_models()

        # Lazy imports for F0 extractor
        from rvc_infer import extract_f0_pyin, extract_f0_autocorr
        # Default: use autocorr for speed; pyin available via --f0-method=pyin
        f0_method = getattr(self, "f0_method", "autocorr")
        f0_func = extract_f0_pyin if f0_method == "pyin" else extract_f0_autocorr

        last_infer = time.time()
        while not self._stop_flag.is_set():
            now = time.time()
            if now - last_infer < self.hop_ms / 1000.0:
                time.sleep(0.01)
                continue
            last_infer = now

            try:
                # Get latest context (most recent context_ms)
                ctx_16k = self.ring_in.read_latest(self.context_samples_16k)
                if np.max(np.abs(ctx_16k)) < 0.005:
                    # Silent input — output silence
                    self.ring_out.write(np.zeros(self.out_hop_samples, dtype=np.float32))
                    continue

                t0 = time.time()
                # HuBERT features
                feats = self._hubert.extract(ctx_16k)
                t_hubert = (time.time() - t0) * 1000

                t1 = time.time()
                # F0 (autocorr for speed; pyin optional)
                f0 = f0_func(ctx_16k, sr=self.feat_sr)
                t_f0 = (time.time() - t1) * 1000

                t2 = time.time()
                # RVC inference
                wav_48k = self._model.convert(feats, f0, f0_up_key=self.f0_up_key)
                t_rvc = (time.time() - t2) * 1000

                t_infer = (time.time() - t0) * 1000
                self._infer_time_ms = t_infer
                self._infer_count += 1

                # wav_48k is at 48kHz. Take the latest hop_ms portion.
                hop_48k = int(self.rvc_sr * self.hop_ms / 1000)
                if len(wav_48k) >= hop_48k:
                    chunk_48k = wav_48k[-hop_48k:]
                else:
                    chunk_48k = np.pad(wav_48k, (hop_48k - len(wav_48k), 0))

                # Resample to device_sr
                if self.device_sr == self.rvc_sr:
                    chunk_out = chunk_48k
                else:
                    chunk_out = self._resample(chunk_48k, self.rvc_sr, self.device_sr)

                # If chunk is shorter than out_hop_samples, pad with zeros
                if len(chunk_out) < self.out_hop_samples:
                    chunk_out = np.pad(chunk_out, (0, self.out_hop_samples - len(chunk_out)))
                elif len(chunk_out) > self.out_hop_samples:
                    chunk_out = chunk_out[:self.out_hop_samples]

                # Soft-clip to avoid pops
                chunk_out = np.tanh(chunk_out * 0.95)
                self.ring_out.write(chunk_out.astype(np.float32))

                if self._infer_count % 3 == 0 or t_infer > self.hop_ms * 1.5:
                    log.info(
                        f"  [infer #{self._infer_count}] total={t_infer:.0f}ms "
                        f"hubert={t_hubert:.0f} f0={t_f0:.0f} rvc={t_rvc:.0f} "
                        f"out_ring={self.ring_out.available()}/{self.output_ring_capacity}"
                    )
            except Exception as e:
                log.exception(f"Inference error: {e}")
                time.sleep(0.1)

        log.info("Inference thread stopped")

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self.ring_in.clear()
        self.ring_out.clear()
        self._infer_count = 0
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        # Wait briefly for models to load
        if not self._ready_event.wait(timeout=30):
            raise RuntimeError("RVC model load timed out")
        log.info(f"RealtimeRVC started: ctx={self.context_ms}ms hop={self.hop_ms}ms")

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("RealtimeRVC stopped")

    # ----------------------------------------------------------------
    # Parameter updates (thread-safe: just assign atomic in CPython)
    # ----------------------------------------------------------------
    def set_f0_up_key(self, key: float):
        self.f0_up_key = float(key)
        log.info(f"f0_up_key → {key:+.1f} 半音")

    # Short aliases for ergonomic use from audio callback
    def push(self, block, block_sr=None):
        if block_sr is None:
            block_sr = self.device_sr
        self.push_audio(block, block_sr)

    def pull(self, n_samples):
        return self.pull_audio(n_samples)

    @property
    def latency_ms(self) -> float:
        return self.context_ms + self.hop_ms

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return x
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        up = sr_out // g
        down = sr_in // g
        return resample_poly(x, up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test: capture from default mic, play through default speaker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sounddevice as sd

    ap = argparse.ArgumentParser(description="Realtime RVC voice changer (CLI)")
    ap.add_argument("-m", "--model", default="models/三月七.pth")
    ap.add_argument("--hubert", default="models/chinese-hubert-base")
    ap.add_argument("--key", type=float, default=12.0,
                    help="F0 shift semitones (default +12 = octave up)")
    ap.add_argument("--context-ms", type=int, default=600)
    ap.add_argument("--hop-ms", type=int, default=200)
    ap.add_argument("--f0-method", choices=["autocorr", "pyin"], default="autocorr",
                    help="F0 extraction method (autocorr=fast, pyin=accurate)")
    ap.add_argument("--in-dev", type=int, default=None, help="input device index")
    ap.add_argument("--out-dev", type=int, default=None, help="output device index")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run (0 = forever)")
    ap.add_argument("--list-devs", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.list_devs:
        print(sd.query_devices())
        raise SystemExit(0)

    in_dev = args.in_dev if args.in_dev is not None else sd.default.device[0]
    out_dev = args.out_dev if args.out_dev is not None else sd.default.device[1]
    in_sr = int(sd.query_devices(in_dev, "input")["default_samplerate"])
    out_sr = int(sd.query_devices(out_dev, "output")["default_samplerate"])

    log.info(f"In  dev {in_dev}: {sd.query_devices(in_dev)['name']} @ {in_sr}Hz")
    log.info(f"Out dev {out_dev}: {sd.query_devices(out_dev)['name']} @ {out_sr}Hz")

    # Use the output SR for both streams (sounddevice handles rate matching)
    sr = out_sr
    blocksize = int(sr * args.hop_ms / 2000)  # half of hop_ms for low callback latency

    engine = RealtimeRVC(
        pth_path=args.model,
        hubert_path=args.hubert,
        device_sr=sr,
        context_ms=args.context_ms,
        hop_ms=args.hop_ms,
        f0_up_key=args.key,
    )
    engine.f0_method = args.f0_method

    log.info("Loading models...")
    engine.start()
    log.info(f"Engine ready. Latency ≈ {engine.latency_ms}ms")

    state = {"frames": 0, "t0": time.time()}

    def cb(indata, outdata, frames, time_info, status):
        if status:
            log.warning(f"Stream status: {status}")
        mono = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)
        engine.push_audio(mono, sr)
        out = engine.pull_audio(frames)
        outdata[:len(out), 0] = out
        if outdata.shape[1] > 1:
            outdata[:, 1] = out
        state["frames"] += frames

    stream = sd.Stream(
        device=(in_dev, out_dev),
        samplerate=sr,
        channels=1,
        blocksize=blocksize,
        dtype="float32",
        callback=cb,
    )
    log.info(f"Streaming. blocksize={blocksize} ({blocksize/sr*1000:.1f}ms)")
    log.info("Press Ctrl+C to stop.")
    try:
        with stream:
            t_end = time.time() + args.duration if args.duration > 0 else None
            while True:
                time.sleep(0.1)
                if t_end and time.time() >= t_end:
                    break
                if state["frames"] % (sr * 2) < blocksize:
                    elapsed = time.time() - state["t0"]
                    avg_infer = engine._infer_time_ms
                    log.info(
                        f"  [{elapsed:.1f}s] frames={state['frames']} "
                        f"infer={avg_infer:.0f}ms"
                    )
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        log.info("Stopped.")
