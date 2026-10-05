"""RVC inference engine for voice conversion.

End-to-end pipeline: wav → HuBERT features → F0 → RVC model → output wav.
Uses the official RVC-Project code from rvc_lib/ for the model architecture.
"""
from __future__ import annotations

import os
import sys
import json
import logging
import numpy as np
import torch
from pathlib import Path

# Make rvc_lib importable
_HERE = Path(__file__).resolve().parent
RVC_LIB = _HERE / "rvc_lib"
if str(RVC_LIB) not in sys.path:
    sys.path.insert(0, str(RVC_LIB))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rvc_infer")


# ---------------------------------------------------------------------------
# F0 extraction (lightweight CPU-only — no parselmouth/rmvpe needed)
# ---------------------------------------------------------------------------

def extract_f0_pyin(wav: np.ndarray, sr: int, frame_period_ms: float = 10.0) -> np.ndarray:
    """Extract F0 contour using librosa.pyin.

    Returns 1-D numpy array of F0 in Hz, with `0` for unvoiced frames.
    Length matches HuBERT feature frames at `frame_period_ms` resolution.
    """
    import librosa
    hop = int(sr * frame_period_ms / 1000)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        wav.astype(np.float32),
        fmin=50,
        fmax=1100,
        sr=sr,
        frame_length=2048,
        hop_length=hop,
        fill_na=0.0,
    )
    # f0 with NaN for unvoiced; replace with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def extract_f0_autocorr(wav: np.ndarray, sr: int, frame_period_ms: float = 10.0) -> np.ndarray:
    """Cheap autocorrelation F0 extractor (fallback if pyin too slow)."""
    import scipy.signal as sp
    hop = int(sr * frame_period_ms / 1000)
    # Pre-emphasize
    emphasized = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    n_frames = 1 + (len(emphasized) - 1024) // hop
    f0 = np.zeros(n_frames, dtype=np.float32)
    win = np.hanning(1024)
    lag_min = int(sr / 1100)  # 1100 Hz
    lag_max = int(sr / 50)    # 50 Hz
    for i in range(n_frames):
        s = emphasized[i * hop : i * hop + 1024] * win
        if np.max(np.abs(s)) < 0.005:  # silent frame
            continue
        ac = np.correlate(s, s, mode="full")[1024:]  # auto correlations [0..inf]
        # find first peak in [lag_min..lag_max]
        peak_idx = lag_min + np.argmax(ac[lag_min:lag_max])
        if peak_idx > 0 and ac[peak_idx] > 0.3 * ac[0]:
            f0[i] = sr / peak_idx
    return f0


# ---------------------------------------------------------------------------
# HuBERT feature extraction (uses official RVC infer.hubert code)
# ---------------------------------------------------------------------------

class HuBERTExtractor:
    def __init__(self, model_path: str, device: str = "cpu"):
        from transformers import AutoFeatureExtractor, HubertModel
        self.device = device
        log.info(f"Loading HuBERT from {model_path}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def extract(self, wav_16k: np.ndarray) -> np.ndarray:
        """Extract HuBERT features from 16kHz mono wav.

        Returns numpy array [T, 768].
        """
        inputs = self.feature_extractor(
            wav_16k.astype(np.float32),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values, attention_mask=attention_mask)
            feats = outputs.last_hidden_state[0]  # [T, 768]
        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# RVC Model wrapper
# ---------------------------------------------------------------------------

class RVCModel:
    """Loads a .pth RVC model and provides conversion API."""

    def __init__(self, pth_path: str, device: str = "cpu"):
        from infer.module.models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs768NSFsid,
        )

        log.info(f"Loading RVC .pth from {pth_path}")
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        self.sr_str = ckpt.get("sr", "48k")
        self.f0 = ckpt.get("f0", 1)
        self.info = ckpt.get("info", "")
        self.spk_id = 0  # single speaker model

        sr2sr = {"32k": 32000, "40k": 40000, "48k": 48000}
        self.sr = sr2sr.get(self.sr_str, 48000)

        cfg = list(ckpt["config"])
        # If config[16] is gin_channels; we use Ms768 since model expected 768 dim emb_phone
        cfg[17] = sr2sr.get(cfg[17], cfg[17]) if isinstance(cfg[17], str) else cfg[17]

        # Detect 768 vs 256: the checkpoint's emb_phone weight tells us
        # Just always try Ms768 first; if shape mismatch fall back to Ms256
        sd = ckpt["weight"]
        # Find emb_phone shape
        emb_shape = None
        for k, v in sd.items():
            if k.endswith("enc_p.emb_phone.weight"):
                emb_shape = tuple(v.shape)
                break
        if emb_shape is None:
            raise RuntimeError("Checkpoint missing enc_p.emb_phone.weight")

        log.info(f"  emb_phone shape: {emb_shape} → "
                 f"{'768-dim' if emb_shape[1] == 768 else '256-dim'}")
        ModelCls = SynthesizerTrnMs768NSFsid if emb_shape[1] == 768 else SynthesizerTrnMs256NSFsid

        self.model = ModelCls(*cfg[:17], cfg[17], is_half=False)
        self.model.eval()

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            log.warning(f"  {len(missing)} missing keys (expected for enc_q): "
                        f"{[k for k in missing if 'enc_q' not in k][:5]}")
        if unexpected:
            log.warning(f"  {len(unexpected)} unexpected keys: {unexpected[:5]}")

        self.device = device
        self.model = self.model.to(device)
        log.info(f"  ✓ Loaded RVC {self.sr_str} model ({self.info}) on {device}")

    @torch.no_grad()
    def convert(
        self,
        phone_features: np.ndarray,
        pitch: np.ndarray,
        f0_up_key: float = 0.0,
        sid: int = 0,
        return_length2: int = None,
    ) -> np.ndarray:
        """Run inference.

        Args:
            phone_features: [T, 768] HuBERT features
            pitch: [F0_frames] F0 in Hz (unvoiced = 0); will be aligned to phone_features T
            f0_up_key: semitones to shift F0 (positive = higher)
            sid: speaker id
        """
        # Resample F0 to phone-feature length if needed
        T = phone_features.shape[0]
        if len(pitch) != T:
            pitch = np.interp(
                np.linspace(0, 1, T),
                np.linspace(0, 1, len(pitch)),
                pitch,
            ).astype(np.float32)

        # Apply f0 shift
        if f0_up_key != 0:
            shift = 2 ** (f0_up_key / 12)
            pitch = pitch * shift

        # Convert F0 to coarse pitch (1 unit = 1/256 octave in log domain)
        # 256 bins cover ~12 semitones × ~21 semitones
        pitchf = np.zeros_like(pitch, dtype=np.float32)
        voiced = pitch > 0
        # RVC uses: coarse = (f0 / 20).log() mapped to 0..255
        # In their hubert.py they compute: coarse = f0_to_coarse(f0)
        # f0_to_coarse(f0) = (f0 - 20) mapped log to 256 bins
        # Actually it's: 255 * (log(f0) - log(20)) / (log(1100) - log(20))
        pitch_coarse = np.zeros_like(pitch, dtype=np.long)
        LOG20 = np.log(20.0)
        LOG1100 = np.log(1100.0)
        pitch_coarse[voiced] = np.clip(
            ((np.log(pitch[voiced]) - LOG20) / (LOG1100 - LOG20) * 255).astype(np.long),
            0, 255,
        )

        # To torch
        phone_t = torch.from_numpy(phone_features).unsqueeze(0).float().to(self.device)  # [1, T, 768]
        phone_lengths = torch.tensor([T], dtype=torch.long).to(self.device)
        pitch_t = torch.from_numpy(pitch_coarse).unsqueeze(0).long().to(self.device)  # [1, T]
        pitchf_t = torch.from_numpy(pitchf).unsqueeze(0).to(self.device)  # [1, T]
        sid_t = torch.tensor([sid], dtype=torch.long).to(self.device)

        out, _, _ = self.model.infer(
            phone_t,
            phone_lengths,
            pitch_t,
            pitchf_t,
            sid_t,
            return_length2=return_length2,
        )
        return out[0, 0].cpu().numpy()  # [samples]


# ---------------------------------------------------------------------------
# High-level convert_file API
# ---------------------------------------------------------------------------

def load_wav_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load wav file, convert to mono float32, resample to target_sr."""
    import librosa
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav.astype(np.float32)


def convert_file(
    input_path: str,
    output_path: str,
    pth_path: str,
    hubert_path: str,
    f0_up_key: float = 0.0,
    sid: int = 0,
    device: str = "cpu",
):
    """Convert a single audio file end-to-end.

    Args:
        input_path: source wav (any sr)
        output_path: where to save converted wav
        pth_path: RVC .pth file
        hubert_path: chinese-hubert-base directory
        f0_up_key: semitones to shift (+12 = one octave up)
        sid: speaker id (single-speaker models = 0)
        device: 'cpu' or 'cuda'
    """
    import soundfile as sf

    log.info(f"[1/4] Load wav: {input_path}")
    wav = load_wav_16k(input_path, target_sr=16000)
    log.info(f"      Loaded {len(wav)/16000:.2f}s @ 16kHz")

    log.info(f"[2/4] HuBERT features")
    hubert = HuBERTExtractor(hubert_path, device=device)
    feats = hubert.extract(wav)
    log.info(f"      Features: {feats.shape}")

    log.info(f"[3/4] F0 extraction (pyin)")
    f0 = extract_f0_pyin(wav, sr=16000, frame_period_ms=20.0)
    # HuBERT features at ~50fps; pyin at 50fps (20ms hop) -> same rate
    log.info(f"      F0 frames: {len(f0)}, voiced: {(f0 > 0).sum()}")

    log.info(f"[4/4] RVC inference")
    rvc = RVCModel(pth_path, device=device)
    out_wav = rvc.convert(feats, f0, f0_up_key=f0_up_key, sid=sid)
    log.info(f"      Output: {len(out_wav)/rvc.sr:.2f}s @ {rvc.sr}Hz, "
             f"peak={np.abs(out_wav).max():.3f}")

    log.info(f"Save → {output_path}")
    sf.write(output_path, out_wav, rvc.sr)
    log.info(f"  ✓ Done")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RVC voice conversion")
    ap.add_argument("-i", "--input", required=True, help="input wav")
    ap.add_argument("-o", "--output", required=True, help="output wav")
    ap.add_argument("-m", "--model", default="models/三月七.pth")
    ap.add_argument("--hubert", default="models/chinese-hubert-base")
    ap.add_argument("--key", type=float, default=0.0, help="f0 shift semitones")
    ap.add_argument("--sid", type=int, default=0)
    args = ap.parse_args()

    convert_file(
        args.input, args.output,
        args.model, args.hubert,
        f0_up_key=args.key,
        sid=args.sid,
    )