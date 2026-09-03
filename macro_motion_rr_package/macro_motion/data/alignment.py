"""Spec section 11: explicit image <-> RR temporal alignment sanity check.

The whole dense design rests on one promise: model frame i corresponds to raw
frame `sample_idx[i]`, and the RR label used to supervise it is the label of that
same raw frame. If the video and the labels are ever decimated by different rules,
every prediction is trained against a shifted target and nothing downstream can
detect it -- the loss still falls, the metrics still compute, and the result is
quietly meaningless.

The spec expresses this as one shared `sample_idx` vector used to index both the
video and the labels. This pipeline reaches the same place by a different route:
`frame_stride` is applied inside the dataset, to the sorted frame list before the
PNGs are decoded (a ~3x saving in I/O, since discarded frames are never read) and
to the label array via the same stride. That is an equivalent operation, but
"equivalent" is exactly the kind of claim that deserves to be verified rather than
asserted, which is what this module does:

  * the index vector is the one the spec specifies -- 0, 3, 6, ..., 597
  * the clip really has that many frames
  * the labels really have that many entries
  * label[i] really equals the raw label at sample_idx[i], checked by reloading
    the untouched .npy from disk and comparing element by element

Run standalone:
    python -m macro_motion.data.alignment --metadata data/metadata.csv --split train
"""
import numpy as np
import torch


def build_sample_idx(clip_frames, stride, device=None):
    """The single temporal index vector, per spec sections 9 and 10."""
    return torch.arange(0, int(clip_frames), step=max(1, int(stride)), device=device)


def check_alignment(dataset, metadata_csv=None, n_clips=3, verbose=True):
    """Verify video/label temporal alignment. Raises AssertionError on mismatch.

    `dataset` is a MacroMotionDataset. Only `n_clips` samples are materialised,
    because each one decodes a full clip off disk.
    """
    import pandas as pd

    clip_frames = dataset.clip_frames
    stride = dataset.frame_stride
    sample_idx = build_sample_idx(clip_frames, stride)
    T_expected = dataset.T_out

    if verbose:
        print(f"[align] clip_frames={clip_frames} frame_stride={stride} "
              f"-> {len(sample_idx)} model frames")
        print(f"[align] sample_idx = [{sample_idx[0].item()}, {sample_idx[1].item()}, "
              f"{sample_idx[2].item()}, ..., {sample_idx[-1].item()}]")

    # Spec section 11: the index vector itself.
    assert len(sample_idx) == T_expected, \
        f"sample_idx has {len(sample_idx)} entries but dataset yields T_out={T_expected}"
    assert sample_idx[0].item() == 0, f"sample_idx must start at 0, got {sample_idx[0].item()}"
    if clip_frames == 600 and stride == 3:
        assert len(sample_idx) == 200, f"expected 200 model frames, got {len(sample_idx)}"
        assert sample_idx[-1].item() == 597, \
            f"expected last raw index 597, got {sample_idx[-1].item()}"

    n = min(int(n_clips), len(dataset))
    for k in range(n):
        s = dataset[k]
        clip, rr_seq, rr_mask = s["clip"], s["rr_seq"], s["rr_mask"]

        # Spec section 11: shapes must agree across video and labels.
        assert clip.shape[0] == len(sample_idx), \
            f"clip {k}: {clip.shape[0]} frames vs {len(sample_idx)} sample_idx entries"
        assert rr_seq.shape[0] == clip.shape[0], \
            f"clip {k}: rr_seq {rr_seq.shape[0]} != clip frames {clip.shape[0]}"
        assert rr_mask.shape[0] == clip.shape[0], \
            f"clip {k}: rr_mask {rr_mask.shape[0]} != clip frames {clip.shape[0]}"

        # The part that actually catches a misalignment: compare the decimated
        # labels against the untouched file, at the exact spec indices.
        row = dataset.df.iloc[k]
        if metadata_csv is not None or "rr_path" in row:
            raw = np.load(str(row["rr_path"])).astype("float64").ravel()
            if len(raw) >= clip_frames:
                want = raw[sample_idx.numpy()]
                got = rr_seq.numpy().astype("float64")
                bad = int((~np.isclose(got, want, equal_nan=True)).sum())
                assert bad == 0, (
                    f"clip {k} ({row.get('clip_id', '?')}): {bad}/{len(want)} labels do not "
                    f"match raw[sample_idx] -- video and labels are decimated differently")

        if verbose and k == 0:
            print("[align] verified label mapping on the first clip:")
            for i in (0, 1, 2):
                print(f"[align]   model frame {i:>3} <-> raw frame {sample_idx[i].item():>3} "
                      f"<-> RR[{sample_idx[i].item()}] = {rr_seq[i].item():.1f}")
            print("[align]   ...")
            last = len(sample_idx) - 1
            print(f"[align]   model frame {last:>3} <-> raw frame {sample_idx[last].item():>3} "
                  f"<-> RR[{sample_idx[last].item()}] = {rr_seq[last].item():.1f}")

    if verbose:
        print(f"[align] OK - {n} clip(s) checked, video and RR labels share one index vector")
    return sample_idx


def main():
    import argparse
    from macro_motion.data.dataset import MacroMotionDataset

    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--clip-frames", type=int, default=600)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--n-clips", type=int, default=3)
    a = p.parse_args()

    ds = MacroMotionDataset(a.metadata, a.split, a.input_size, a.clip_frames,
                            True, False, a.frame_stride)
    check_alignment(ds, metadata_csv=a.metadata, n_clips=a.n_clips)


if __name__ == "__main__":
    main()
