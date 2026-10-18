# -*- coding: utf-8 -*-
"""IndexTTS-2.5 CLI smoke test.

用法（在项目根目录）:
  _indextts_venv\\Scripts\\python.exe ind_extts_cli.py ^
      --prompt-wav "models\\gsv\\v2ProPlus\\三月七\\reference_audios\\中文\\emotions\\【开心】....wav" ^
      --text "你好，这是 IndexTTS 的声音克隆测试。" ^
      --lang ZH ^
      --output tts_outputs\\sanyueqi_test.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '_indextts_src')
CHECKPOINTS = os.path.join(SRC, 'checkpoints')

# 让 indextts 包可导入
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# 辅助模型缓存到 checkpoints/hf_cache（infer_v2_5 默认查找位置）
os.environ.setdefault('HF_HUB_CACHE', os.path.join(CHECKPOINTS, 'hf_cache'))
# ModelScope 缓存放项目内，避免 C:\Users\...\.cache\modelscope 权限问题
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(HERE, '_ms_cache'))
os.makedirs(os.environ['MODELSCOPE_CACHE'], exist_ok=True)
# 强制走 ModelScope（HF 直连在国内不可用）
os.environ.setdefault('USE_MODELSCOPE', 'true')
# 用户主目录 AppData 写入受限 -> 把各类缓存都挪到项目内
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_cache', 'matplotlib'))
os.environ.setdefault('HF_HOME', os.path.join(HERE, '_cache', 'hf'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(HERE, '_cache'))
for _d in (os.environ['MPLCONFIGDIR'], os.environ['HF_HOME'], os.environ['XDG_CACHE_HOME']):
    os.makedirs(_d, exist_ok=True)
# CPU 线程：默认用满所有逻辑核（torch 默认只用物理核数的一半）
os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count() or 8))
os.environ.setdefault('MKL_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
os.environ.setdefault('KMP_BLOCKTIME', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    ap = argparse.ArgumentParser(description='IndexTTS-2.5 CLI')
    ap.add_argument('--prompt-wav', required=True, help='参考音频（3-15 秒）')
    ap.add_argument('--text', required=True, help='要合成的文本')
    ap.add_argument('--lang', default='ZH',
                    choices=['ZH', 'EN', 'JA', 'ES', 'AR'], help='语言')
    ap.add_argument('--output', default='tts_outputs/indextts_out.wav')
    ap.add_argument('--duration-factor', type=float, default=1.0,
                    help='语速：>1 变慢，<1 变快，范围 0.5-2.0')
    ap.add_argument('--no-qwen', action='store_true',
                    help='不加载 Qwen 情感模型（省内存，但 use_emo_text 不可用）')
    # ---- CPU 速度调优 ----
    ap.add_argument('--threads', type=int, default=0,
                    help='torch CPU 线程数，0=自动(物理核数)')
    ap.add_argument('--num-beams', type=int, default=1,
                    help='GPT 采样 beam 数，默认 1（最快）；3 是上游默认')
    ap.add_argument('--diffusion-steps', type=int, default=20,
                    help='s2mel 扩散步数，默认 20（上游 25；扫描显示 20 无削波，'
                         '15 以下 peak 逼近 1.0）')
    ap.add_argument('--cfg-rate', type=float, default=0.7,
                    help='s2mel CFG rate，默认 0.7')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=30)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--peak-limit', type=float, default=0.95,
                    help='写盘前峰值上限，超过则整体缩放（默认 0.95）；'
                         '设 1.0 关闭')
    args = ap.parse_args()

    if not os.path.isdir(CHECKPOINTS):
        sys.exit(f'[err] checkpoints 不存在: {CHECKPOINTS}')
    cfg = os.path.join(CHECKPOINTS, 'config.yaml')
    if not os.path.isfile(cfg):
        sys.exit(f'[err] config.yaml 不存在: {cfg}')

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ---- torch 线程调优 ----
    import torch
    if args.threads > 0:
        nthreads = args.threads
    else:
        nthreads = os.cpu_count() or 8
    torch.set_num_threads(nthreads)
    try:
        torch.set_num_interop_threads(max(1, nthreads // 2))
    except RuntimeError:
        pass  # 已初始化时设置会抛错，忽略

    print(f'[cli] python      = {sys.version.split()[0]}')
    print(f'[cli] src         = {SRC}')
    print(f'[cli] checkpoints = {CHECKPOINTS}')
    print(f'[cli] prompt_wav  = {os.path.basename(args.prompt_wav)}')
    print(f'[cli] text        = {args.text[:50]}{"..." if len(args.text) > 50 else ""}')
    print(f'[cli] lang        = {args.lang}')
    print(f'[cli] output      = {args.output}')
    print(f'[cli] cpu_threads = {nthreads}  beams={args.num_beams}  '
          f'diff_steps={args.diffusion_steps}  cfg={args.cfg_rate}')

    # 覆盖推理超参（s2mel 扩散步数 / CFG）
    import indextts.infer_v2_5 as iv25
    iv25.DIFFUSION_STEPS = args.diffusion_steps
    iv25.CFG_RATE = args.cfg_rate
    from indextts.infer_v2_5 import IndexTTS2

    t0 = time.time()
    print('[cli] loading model (CPU, 可能需要几分钟)...')
    tts = IndexTTS2(
        cfg_path=cfg,
        model_dir=CHECKPOINTS,
        use_bf16=True,          # CPU 下会被自动关掉
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_qwen_emo=not args.no_qwen,
    )
    print(f'[cli] model loaded in {time.time() - t0:.1f}s')

    t1 = time.time()
    print('[cli] synthesizing...')
    tts.infer(
        spk_audio_prompt=args.prompt_wav,
        text=args.text,
        lang=args.lang,
        output_path=args.output,
        duration_factor=args.duration_factor,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        verbose=False,
    )
    dt = time.time() - t1
    print(f'[cli] synthesis done in {dt:.1f}s')
    if not os.path.isfile(args.output):
        sys.exit('[err] 输出文件未生成')

    # 峰值保护：不同参考音频的响度差异会让输出电平漂移，
    # 实测青雀的某个 take 峰值顶到 1.000 有削波。超过阈值时整体缩放，
    # 不改变频谱和音色，只是把电平拉回安全区。
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(args.output, dtype='float64')
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak > args.peak_limit:
        data = data * (args.peak_limit / peak)
        sf.write(args.output, data, sr, subtype='PCM_16')
        print(f'[cli] peak {peak:.3f} > {args.peak_limit:.2f}, '
              f'scaled by {args.peak_limit / peak:.4f}')

    data, sr = sf.read(args.output)
    dur = len(data) / sr if sr else 0
    final_peak = float(np.abs(data).max()) if len(data) else 0.0
    print(f'[cli] OK -> {args.output}  sr={sr}  dur={dur:.2f}s  '
          f'peak={final_peak:.3f}  RTF={dt / max(dur, 1e-6):.3f}')


if __name__ == '__main__':
    main()
