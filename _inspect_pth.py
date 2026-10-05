"""Detailed inspection of RVC .pth weight dict."""
import sys
import torch


def main():
    pth_path = sys.argv[1]
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)

    print(f"=== Top-level summary ===")
    print(f"  Type: {type(ckpt).__name__}")
    print(f"  Keys: {list(ckpt.keys())}")

    print(f"\n=== Config (raw list) ===")
    cfg = ckpt["config"]
    print(f"  Config has {len(cfg)} values:")
    for i, v in enumerate(cfg):
        print(f"    [{i:2d}] {v!r}")

    print(f"\n=== Info / sr / f0 / version ===")
    for k in ("info", "sr", "f0", "version"):
        v = ckpt.get(k)
        print(f"  {k}: {v!r}")

    print(f"\n=== Weight keys (first 80) ===")
    sd = ckpt["weight"]
    keys = list(sd.keys())
    print(f"  Total keys: {len(keys)}")
    for k in keys[:80]:
        v = sd[k]
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {tuple(v.shape)}")
        else:
            print(f"    {k}: {type(v).__name__}")
    if len(keys) > 80:
        print(f"  ... ({len(keys) - 80} more)")

    # Group by prefix
    print(f"\n=== Weight prefix groups ===")
    prefixes = set()
    for k in keys:
        parts = k.split(".")
        prefix = ".".join(parts[:1]) if len(parts) > 1 else k
        prefixes.add(prefix)
    for p in sorted(prefixes):
        count = sum(1 for k in keys if k.startswith(p + ".") or k == p)
        print(f"  {p}: {count} keys")


if __name__ == "__main__":
    main()