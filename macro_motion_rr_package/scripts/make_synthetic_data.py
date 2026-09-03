"""
Generate synthetic NIR clips + metadata.csv so the full RR pipeline can be
exercised end-to-end without real NICU data.

Each clip is a (T, H, W, 3) uint8 .npy whose global brightness oscillates at a
known breathing frequency; the RR label (breaths/min) is derived from that
frequency, so a working model has a real (if easy) signal to fit.

Usage:
  python scripts/make_synthetic_data.py \
      --out_dir data/synthetic --metadata data/metadata.csv \
      --subjects 6 --clips_per_subject 3 --frames 60 --size 64 --fps 30
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_clip(rr_bpm, frames, size, fps, rng):
    # breathing frequency in Hz from breaths-per-minute
    f = rr_bpm / 60.0
    t = np.arange(frames) / fps
    # global brightness oscillation + a moving soft blob (the "chest")
    base = 90.0 + 50.0 * np.sin(2 * np.pi * f * t)  # (T,)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = size / 2.0, size / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    blob = np.exp(-r2 / (2 * (size / 4.0) ** 2))  # (H,W)
    clip = np.empty((frames, size, size, 3), dtype=np.uint8)
    for i in range(frames):
        frame = base[i] * (0.5 + 0.5 * blob)
        frame = frame + rng.normal(0, 4.0, size=(size, size))  # sensor noise
        frame = np.clip(frame, 0, 255)
        clip[i] = np.repeat(frame[..., None], 3, axis=2).astype(np.uint8)
    return clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/synthetic")
    p.add_argument("--metadata", default="data/metadata.csv")
    p.add_argument("--subjects", type=int, default=6)
    p.add_argument("--clips_per_subject", type=int, default=3)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for s in range(args.subjects):
        subject_id = f"subj{s:02d}"
        for c in range(args.clips_per_subject):
            # NICU infant RR is roughly 30-60 bpm
            rr = float(rng.uniform(30, 60))
            clip = make_clip(rr, args.frames, args.size, args.fps, rng)
            clip_path = out_dir / f"{subject_id}_clip{c}.npy"
            np.save(clip_path, clip)
            cond = "exposed" if rng.random() < 0.5 else "blanket"
            rows.append({
                "clip_path": str(clip_path),
                "rr_bpm": round(rr, 2),
                "subject_id": subject_id,
                "cond": cond,
            })

    df = pd.DataFrame(rows)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.metadata, index=False)
    print(df)
    print(f"\nWrote {len(df)} clips to {out_dir} and metadata to {args.metadata}")


if __name__ == "__main__":
    main()
