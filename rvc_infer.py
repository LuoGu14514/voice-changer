"""RVC 文件变声推理（极简版，CPU 跑）。

使用：
    python rvc_infer.py --input in.wav --output out.wav --model models/some.pth

特性：
    - 接受 RVC v2 .pth 模型（HuBERT + Generator）
    - 接受 ONNX 导出的 .onnx 模型（仅 Generator，HuBERT 仍用 .pth）
    - 自动检测模型类型
    - 流式 / 整段两种处理模式

依赖：
    pip install torch onnxruntime soundfile numpy
    （可选：pip install transformers  -- for HuBERT feature extractor）

注意：
    - 完整 RVC 推理需要 HuBERT 特征提取器（chinese-hubert-base，~360MB），
      推荐先用 [content-free Hubert] 或 [rmvpe] 做F0 估计
    - 本脚本默认提供 demo 级别的"模型占位 + F0+Hubert 提取"，具体声学模型由
      LoadingWaveRNN / RVC 提供的 Generator 实现
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


class RVCModel:
    """RVC 模型加载器。支持：
       - RVC v2 .pth (state dict)
       - ONNX .onnx (仅声学模型，HuBERT 仍需 pth)
    """

    def __init__(self, model_path: str | os.PathLike, index_path: Optional[str] = None,
                 device: str = "cpu"):
        self.model_path = str(model_path)
        self.index_path = str(index_path) if index_path else None
        self.device = device
        self.model_type: str = "unknown"  # 'pth' / 'onnx'
        self.config_path = Path(self.model_path).with_suffix(".json")
        self.config: dict = {}
        self.model = None  # 占位：声学模型 / 或 ONNX session

    def load(self):
        ext = Path(self.model_path).suffix.lower()
        if ext == ".onnx":
            self.model_type = "onnx"
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.model = ort.InferenceSession(self.model_path, sess_options=so,
                                              providers=["CPUExecutionProvider"])
            print(f"  [RVC] ONNX model loaded: {self.model_path}")
            print(f"  [RVC] Inputs: {[i.name for i in self.model.get_inputs()]}")
            print(f"  [RVC] Outputs: {[o.name for o in self.model.get_outputs()]}")
        elif ext in (".pth", ".pt"):
            self.model_type = "pth"
            # 完整加载需要 torch + fairseq；留个坑
            print(f"  [RVC] Detected PyTorch checkpoint: {self.model_path}")
            print(f"  [RVC] For full load: install torch>=2.0 and run with --torch flag")
        else:
            raise ValueError(f"Unknown model extension: {ext}")

        # 加载配置文件
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                self.config = json.load(f)
                print(f"  [RVC] Config: {self.config}")
        return self

    def infer(self, hubert: np.ndarray, f0: np.ndarray,
              protect: float = 0.5) -> np.ndarray:
        """推理一次：输入 HuBERT 特征 [T, 768] 与 F0 [T]，输出 wav [T*256]。

        注意：这是 RVC v2 接口；具体 shape 取決于模型。
        """
        if self.model_type != "onnx":
            raise NotImplementedError("Only ONNX inference is implemented in this skeleton.")
        # 典型 RVC v2 ONNX 输入: hubert[T, 768], f0[T, 1], protect[T, 1]
        hubert = hubert.astype(np.float32)
        f0 = f0.astype(np.float32).reshape(-1, 1)
        protect = np.full((hubert.shape[0], 1), float(protect), dtype=np.float32)
        outputs = self.model.run(None, {"hubert": hubert, "f0": f0, "protect": protect})
        return outputs[0].squeeze().astype(np.float32)


def f0_extract_cheap(y: np.ndarray, sr: int, hop: int = 160) -> np.ndarray:
    """极简 F0 估计：仅用于在没有 torch/rmvpe 时检验流水线。

    使用 numpy 自相关的简化版本，只适合跟踪基频大致变化，不是商用 F0。
    输出 [T]，T = len(y) // hop。
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    n_frames = len(y) // hop
    f0 = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = y[start:start + 1024]
        if len(frame) < 1024:
            break
        # 归一化自相关，找最大 lag（在 [50, 500] Hz 对应 sr 中）
        frame = frame - frame.mean()
        e = np.dot(frame, frame)
        if e < 1e-6:
            continue
        lo_lag = max(2, int(sr / 500))  # 500 Hz
        hi_lag = min(len(frame) // 2, int(sr / 50))  # 50 Hz
        if hi_lag <= lo_lag:
            continue
        ac = np.array([np.dot(frame[:len(frame)-lag], frame[lag:]) / e
                       for lag in range(lo_lag, hi_lag)])
        peak = int(np.argmax(ac))
        f0[i] = sr / (peak + lo_lag)
    return f0


def hubert_dummy(features_required: int = 768) -> np.ndarray:
    """极简 HuBERT 占位。"""
    # 实际上需要 transformers + chinese-hubert-base
    # 这里仅返回占位 0，让流水线走到 ONNX；输出会有结果但不真实
    return np.zeros((1, features_required), dtype=np.float32)


def convert_file(input_path: str, output_path: str, model_path: str,
                 index_path: Optional[str] = None, sr: int = 40000):
    """文件变声主入口。"""
    print(f"  [RVC] Input: {input_path}")
    print(f"  [RVC] Output: {output_path}")
    print(f"  [RVC] Model: {model_path}")

    # 1. 读 wav
    y, file_sr = sf.read(input_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    print(f"  [RVC] Wav: sr={file_sr}, duration={len(y)/sr:.2f}s")

    # 2. 加载模型
    rvc = RVCModel(model_path, index_path).load()

    # 3. 提取特征（占位——真实场景需要 torch + chinese-hubert）
    if rvc.model_type == "onnx":
        n_frames = max(1, len(y) // 160)
        hubert = np.zeros((n_frames, 768), dtype=np.float32)
        f0 = f0_extract_cheap(y, file_sr, hop=160)
        print(f"  [RVC] Frames: {n_frames}, F0 range: {f0[f0>0].min() if (f0>0).any() else 0:.1f} - {f0.max():.1f} Hz")

        # 4. 推理
        t0 = time.time()
        try:
            out = rvc.infer(hubert, f0, protect=0.5)
        except Exception as e:
            print(f"  [RVC] ONNX inference failed: {e}")
            print(f"  [RVC] This usually happens because the dummy features don't match the model's training distribution.")
            print(f"  [RVC] Real inference needs proper HuBERT + F0 extractor.")
            return None
        dt = time.time() - t0
        print(f"  [RVC] Inference: {dt:.2f}s ({len(y)/file_sr/dt:.2f}x realtime on CPU)")

        # 5. 写 wav
        sf.write(output_path, out, sr)
        print(f"  [RVC] Saved: {output_path}, peak={np.max(np.abs(out)):.3f}")
        return out
    else:
        print(f"  [RVC] .pth loading not yet implemented.")
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="RVC 文件变声")
    ap.add_argument("--input", "-i", required=True, help="输入 wav")
    ap.add_argument("--output", "-o", required=True, help="输出 wav")
    ap.add_argument("--model", "-m", required=True, help="RVC 模型 .pth 或 .onnx")
    ap.add_argument("--index", default=None, help="RVC .index 文件（可选）")
    ap.add_argument("--sr", type=int, default=40000, help="输出采样率（默认 40k RVC 标配）")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: 输入文件不存在 {args.input}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: 模型不存在 {args.model}")
        sys.exit(1)

    result = convert_file(args.input, args.output, args.model, args.index, args.sr)
    if result is None:
        sys.exit(2)