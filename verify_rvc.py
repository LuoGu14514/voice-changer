"""验证 rvc_infer 流程（无需 RVC 模型）。

跑：
    python _verify_rvc.py

作用：
    - 检查依赖（numpy, soundfile, onnxruntime）
    - 读一个 wav，提取 F0 + hubert 形状
    - 模拟 RVC pipeline 输出："如果你有 .onnx 模型，它会在这一步推理"
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    print("=== RVC pipeline verification ===\n")

    # 1. 检查依赖
    print("[1] Dependencies:")
    print(f"  numpy: {np.__version__}")
    print(f"  soundfile: {sf.__version__}")
    try:
        import onnxruntime as ort
        print(f"  onnxruntime: {ort.__version__}")
        print(f"  providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"  onnxruntime: NOT INSTALLED ({e})")
        print(f"  Install: pip install onnxruntime")
        sys.exit(1)

    # 2. 读一个测试 wav
    print("\n[2] Read test wav:")
    test_wav = Path("test.wav")
    if not test_wav.exists():
        # fallback: 生成一个 3 秒 440Hz 正弦
        print("  test.wav missing,生成 3 秒 440Hz 正弦代替")
        sr = 48000
        t = np.arange(sr * 3) / sr
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    else:
        y, sr = sf.read(str(test_wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        print(f"  Read: {test_wav}, sr={sr}, duration={len(y)/sr:.2f}s")

    print(f"  shape: {y.shape}, peak={np.max(np.abs(y)):.3f}")

    # 3. 提取 F0
    print("\n[3] F0 extraction (cheap autocoor):")
    t0 = time.time()
    from rvc_infer import f0_extract_cheap
    f0 = f0_extract_cheap(y, sr, hop=160)
    dt = time.time() - t0
    nz = f0[f0 > 0]
    if len(nz) > 0:
        print(f"  {len(f0)} frames, F0: min={nz.min():.1f} Hz, max={nz.max():.1f} Hz, "
              f"median={np.median(nz):.1f} Hz")
    else:
        print(f"  {len(f0)} frames, F0: no valid F0 detected")
    print(f"  Elapsed: {dt:.2f}s")

    # 4. 模拟 HuBERT 占位
    print("\n[4] HuBERT feature shape (real one needs torch + chinese-hubert-base):")
    n_frames = max(1, len(y) // 160)
    print(f"  would have shape: ({n_frames}, 768)")
    print(f"  占位 zeros 代替")

    # 5. 尝试找 RVC 模型
    print("\n[5] Look for RVC models:")
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"  models/ not present. 请自己创建目录并下模型。")
        print(f"  例如: mkdir models")
    else:
        onnx_models = list(models_dir.glob("*.onnx"))
        pth_models = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
        idx_files = list(models_dir.glob("*.index"))
        print(f"  ONNX models: {len(onnx_models)}")
        for p in onnx_models[:5]:
            print(f"    - {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  PyTorch checkpoints: {len(pth_models)}")
        for p in pth_models[:5]:
            print(f"    - {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Index files: {len(idx_files)}")
        for p in idx_files[:5]:
            print(f"    - {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")

        if onnx_models:
            from rvc_infer import RVCModel
            rvc = RVCModel(onnx_models[0]).load()
            print(f"\n  [OK] Model loaded, ready to infer!")
            print(f"   try: python rvc_infer.py -i test.wav -o out.wav -m {onnx_models[0]}")
        elif pth_models:
            print(f"\n  .pth 需要 torch >= 2.0 才能跑")
            print(f"   pip install torch")
        else:
            print(f"\n  尚无模型 —— 请自行下载一个中文女声 RVC 模型")
            print(f"   名字建议: chinese-female-generic.onnx")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()