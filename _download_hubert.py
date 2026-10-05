"""Try downloading chinese-hubert-base from ModelScope."""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Try various known model IDs
candidates = [
    "TencentGameMate/chinese-hubert-base",
    "pengzhendong/chinese-hubert-base",
    "qinxiu/chinese-hubert-base",
    "xuhuishangzhou/chinese-hubert-base",
    "amphicyon/chinese-hubert-base",
    "iic/chinese-hubert-base",
]

import socket
socket.setdefaulttimeout(30)

for mid in candidates:
    print(f"\n=== Trying {mid} ===")
    try:
        from modelscope import snapshot_download
        path = snapshot_download(
            mid,
            cache_dir="models/_hubert_dl",
            allow_file_pattern=["*.json", "*.bin", "*.txt", "*.safetensors", "pytorch_model.bin", "*.model"],
        )
        print(f"  ✓ Downloaded to: {path}")
        import os
        for f in os.listdir(path):
            sz = os.path.getsize(os.path.join(path, f))
            print(f"    - {f}  ({sz:,} bytes)")
        break
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")
        continue