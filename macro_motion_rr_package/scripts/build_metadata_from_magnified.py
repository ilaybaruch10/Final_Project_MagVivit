"""
Build a metadata CSV for the RR pipeline from magnified clips.

For each vid_N under the source record, points:
  clip_path    -> the magnified frame folder (deepmag output)
  rr_path      -> per-frame ground-truth RR sequence (NaNs filled)
  rr_mask_path -> per-frame validity mask, 1 = train/score on this frame, 0 = ignore

Label QC
--------
Rather than discarding a clip that has some bad labels, this script emits a
per-frame mask so training and evaluation use every trustworthy frame and skip
the rest. Two independent defects are marked invalid:

1. NaN labels. The upstream ozsa pipeline leaves NaNs in signs/rr.npy, always
   as *leading* NaNs (frames before the monitor OCR's first successful read):
   ocr.TemporalVoter returns None only while its buffer is empty, and
   session.clean_series interpolates with limit_area="inside", which by
   definition never back-fills a leading gap.

2. Stale voter output. TemporalVoter emits mode() over its last <=30
   *successful* reads, so across a long run of OCR misses it keeps returning a
   stale value. Those frames are not NaN but are fabricated, and no NaN check
   will ever find them. A frame is marked invalid when more than
   --max-stale-frames have elapsed since the last successful read, measured
   from the f####_miss.png debug crops in signs/rr/.

3. Physiologically impossible values. A third OCR failure mode drops a digit,
   turning 40 into 4 and 50 into 5. Those frames are neither NaN nor stale --
   the read "succeeded" -- so checks 1 and 2 both pass them through. Measured
   over the nine magnified infants, every out-of-range value was low (4-14 bpm,
   0.67% of frames overall, 3.96% for the worst infant) and clustered on 4/5/6,
   and 17 of the 102 contiguous runs lasted more than 30 frames, the longest 489
   (~16 s at 30 fps). A sustained 4 bpm on a neonatal monitor is a misread, not
   a measurement, so a spike filter would not have caught these either.
   Controlled by --rr-min/--rr-max.

Labels are ALWAYS NaN-filled even where masked, because IEEE 0 * NaN = NaN --
masking alone would not stop a NaN from reaching the optimizer.

Cleaned labels and masks are written to --clean-labels-dir; the original
signs/rr.npy is never modified (it is shared with the ozsa pipeline).

Usage:
  python scripts/build_metadata_from_magnified.py \
      --src /mnt/pag_disk/pre_magnification/Patient_0/Record_0 \
      --mag_out /mnt/pag_disk/deepmag_work/output \
      --amp 8 \
      --mode temporal_butter_fl0.2_fh2.0_fs30.0_n1_butter \
      --subject Patient_0 \
      --out data/metadata_amp8_temporal.csv
"""
import argparse, os, csv, glob
import numpy as np


def frames_since_last_read(src, vid, n_frames):
    """Per-frame count of frames elapsed since the last successful OCR read.

    Read from the signs/rr debug crops, where a failed read is named
    f####_miss.png. Frames before the very first successful read get inf.
    Returns None when the debug directory is absent, so the caller can skip the
    staleness check rather than silently treating "unknown" as "fresh".
    """
    crop_dir = os.path.join(src, vid, "signs", "rr")
    if not os.path.isdir(crop_dir):
        return None
    crops = [f for f in os.listdir(crop_dir) if f.endswith(".png")]
    if not crops:
        return None
    ok = set(int(f[1:5]) for f in crops if not f.endswith("_miss.png"))

    since = np.full(n_frames, np.inf, dtype=np.float32)
    last = None
    for i in range(n_frames):
        if i in ok:
            last = i
        if last is not None:
            since[i] = i - last
    return since


def fill_nans(rr):
    """Fill NaNs by interpolation; np.interp clamps at the edges.

    Leading NaNs take the first valid value, trailing NaNs the last, and
    interior gaps are bridged linearly. Returns None if nothing is valid.
    """
    valid = ~np.isnan(rr)
    if not valid.any():
        return None
    out = rr.copy()
    idx = np.arange(len(rr))
    out[~valid] = np.interp(idx[~valid], idx[valid], rr[valid])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="record dir containing vid_*/signs/rr.npy")
    p.add_argument("--mag_out", required=True, help="deepmag output dir")
    p.add_argument("--amp", default="10")
    p.add_argument("--mode", default="no",
                   help="output-dir suffix after amp<N>_. 'no'=static, 'yes'=dynamic, or the "
                        "full temporal suffix e.g. temporal_butter_fl0.2_fh2.0_fs30.0_n1_butter")
    p.add_argument("--subject", default="Patient_0")
    p.add_argument("--record", default="Record_0",
                   help="which record of the patient this is. Magnified clips are named "
                        "<subject>_<record>_vid_<N>_frames_..., so a patient with two "
                        "recordings needs one invocation per record. Default keeps the "
                        "single-record behaviour this script started with.")
    p.add_argument("--vid-suffix", default="",
                   help="suffix on the magnified video name, between vid_<N> and _frames. "
                        "deep_motion_mag names output <video_name>_frames_amp<A>_temporal_<filter>, "
                        "so an input named Patient_0_Record_0_vid_0_face needs --vid-suffix _face")
    p.add_argument("--out", required=True)
    p.add_argument("--max-stale-frames", type=int, default=30,
                   help="mask frames more than this many frames after the last successful OCR "
                        "read. Default 30 matches the TemporalVoter window and the upstream "
                        "clean_series max_interp_gap. Set to -1 to disable the staleness check "
                        "and mask on NaN only.")
    p.add_argument("--rr-min", type=float, default=15.0,
                   help="mask frames whose label is below this (bpm). Default 15 sits under "
                        "any real neonatal rate but above the digit-drop artefacts, which "
                        "measured 4-14 bpm. Set to 0 to disable.")
    p.add_argument("--rr-max", type=float, default=120.0,
                   help="mask frames whose label is above this (bpm). No measured value ever "
                        "exceeded it; it is here so the range check is symmetric.")
    p.add_argument("--min-valid-frames", type=int, default=30,
                   help="drop a clip only if fewer than this many frames survive masking. A clip "
                        "still costs a full 600-frame forward pass regardless of how few frames "
                        "are supervised, so near-empty clips are not worth their compute.")
    p.add_argument("--clean-labels-dir", default="data/labels_clean",
                   help="where filled labels and masks are written (originals are never modified)")
    args = p.parse_args()

    rows, missing, rejected = [], [], []
    vids = sorted(
        [d for d in os.listdir(args.src) if d.startswith("vid_")],
        key=lambda s: int(s.split("_")[1]),
    )
    for v in vids:
        n = v.split("_")[1]
        clip_dir = os.path.join(
            args.mag_out,
            f"{args.subject}_{args.record}_vid_{n}{args.vid_suffix}_frames_amp{args.amp}_{args.mode}",
        )
        rr_path = os.path.join(args.src, v, "signs", "rr.npy")
        n_frames = len(glob.glob(os.path.join(clip_dir, "*.png")))
        if n_frames < 1 or not os.path.exists(rr_path):
            missing.append((v, n_frames, os.path.exists(rr_path)))
            continue

        rr = np.load(rr_path).astype(np.float32).flatten()
        nan_mask = np.isnan(rr)

        # Frame is valid unless its label is NaN or the voter had gone stale.
        valid = ~nan_mask
        stale_masked = 0
        if args.max_stale_frames >= 0:
            since = frames_since_last_read(args.src, v, len(rr))
            if since is not None:
                fresh = since <= args.max_stale_frames
                stale_masked = int((valid & ~fresh).sum())
                valid &= fresh

        # NaN compares False against both bounds, so already-invalid frames are
        # not counted twice here.
        with np.errstate(invalid="ignore"):
            in_range = (rr >= args.rr_min) & (rr <= args.rr_max)
        range_masked = int((valid & ~in_range).sum())
        valid &= in_range

        n_valid = int(valid.sum())
        if n_valid < args.min_valid_frames:
            rejected.append((v, f"only {n_valid} valid frames after masking "
                                f"(minimum {args.min_valid_frames})"))
            continue

        # Fill even masked positions: IEEE 0 * NaN = NaN, so masking alone
        # would not keep a NaN out of the loss.
        filled = fill_nans(rr)
        if filled is None:
            rejected.append((v, "every RR label is NaN"))
            continue

        os.makedirs(os.path.abspath(args.clean_labels_dir), exist_ok=True)
        stem = os.path.join(os.path.abspath(args.clean_labels_dir),
                            f"{args.subject}_{args.record}_{v}")
        out_rr_path = f"{stem}_rr.npy"
        out_mask_path = f"{stem}_rrmask.npy"
        np.save(out_rr_path, filled.astype(np.float32))
        np.save(out_mask_path, valid.astype(np.float32))

        # Hard guard: a NaN must never reach training, where it fails silently.
        if np.isnan(filled).any():
            raise RuntimeError(f"{v}: NaNs survived filling - refusing to emit this row")

        rows.append({
            "clip_id": f"{args.subject}_{args.record}_{v}",
            "clip_path": clip_dir,
            "rr_path": out_rr_path,
            "rr_mask_path": out_mask_path,
            "subject_id": args.subject,
            "cond": "exposed",
            "rr_valid_frac": round(n_valid / len(rr), 4),
            "rr_nan_masked": int(nan_mask.sum()),
            "rr_stale_masked": stale_masked,
            "rr_range_masked": range_masked,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = ["clip_id", "clip_path", "rr_path", "rr_mask_path", "subject_id", "cond",
                  "rr_valid_frac", "rr_nan_masked", "rr_stale_masked", "rr_range_masked"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total_frames = sum(600 for _ in rows)
    masked = sum(r["rr_nan_masked"] + r["rr_stale_masked"] + r["rr_range_masked"]
                 for r in rows)
    partial = [r for r in rows if r["rr_valid_frac"] < 1.0]
    print(f"Wrote {args.out}: {len(rows)} clips")
    if total_frames:
        print(f"  masked {masked} frames ({masked/total_frames:.2%} of emitted data); "
              f"{len(rows) - len(partial)}/{len(rows)} clips fully valid")
    for r in sorted(partial, key=lambda r: r["rr_valid_frac"]):
        print(f"  {r['clip_id']}: {r['rr_valid_frac']:.1%} valid "
              f"(NaN {r['rr_nan_masked']}, stale {r['rr_stale_masked']}, "
              f"out-of-range {r['rr_range_masked']})")
    if rejected:
        print(f"Rejected {len(rejected)}:")
        for v, reason in rejected:
            print(f"  {v}: {reason}")
    if missing:
        print(f"Skipped {len(missing)} (no magnified frames or no rr.npy):")
        for v, nf, hasrr in missing:
            print(f"  {v}: frames={nf} rr.npy={hasrr}")


if __name__ == "__main__":
    main()
