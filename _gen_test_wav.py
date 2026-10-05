"""Generate a 3-second test audio (sine sweep 200→500Hz + 1/f noise) for RVC smoke test."""
import numpy as np
import soundfile as sf

sr = 16000
T = 3.0
t_arr = np.linspace(0, T, int(sr * T), endpoint=False)
# Linear frequency sweep 200->500Hz
freq = 200 + (500 - 200) * t_arr / T
phase = 2 * np.pi * np.cumsum(freq) / sr
sine = 0.4 * np.sin(phase)

# Slight pink-noise modulation (vocal-like variation)
np.random.seed(42)
noise = 0.05 * np.random.randn(t_arr.shape[0])
# Filter to pink-ish (1/f)
from scipy.signal import lfilter
b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1.0, -2.748836033, 2.528730087, -0.777981437]
pink = lfilter(b, a, noise).astype(np.float32)
pink = pink / (np.abs(pink).max() + 1e-9) * 0.2

wav = (sine + pink).astype(np.float32) * 0.7
sf.write('models/test_input.wav', wav, sr)
print(f"Wrote models/test_input.wav: {len(wav)/sr:.2f}s @ {sr}Hz, peak={np.abs(wav).max():.3f}")