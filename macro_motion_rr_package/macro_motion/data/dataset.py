import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from macro_motion.utils.io import load_clip, normalize_to_01, load_mask

COND_TO_ID = {"exposed": 0, "blanket": 1}
# For exposed/partial/heavy use:
# COND_TO_ID = {"exposed": 0, "blanket_partial": 1, "blanket_heavy": 2}
# and change cond_embed in transformer.py to nn.Embedding(3, d_model)

class MacroMotionDataset(Dataset):
    """
    Required CSV columns:
      clip_path, split

    RR label options:
      1) rr_path: .npy/.npz/.csv/.txt sequence, length 600 or similar
      2) rr_seq: text sequence, e.g. "42;42;43;..."
      3) rr_bpm: scalar, repeated across all frames
    """
    def __init__(self, metadata_csv, split="train", input_size=512,
                 clip_frames=600, normalize=True, include_temp=False, frame_stride=1):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split}")
        self.input_size = input_size
        self.clip_frames = clip_frames
        self.normalize = normalize
        self.include_temp = include_temp
        # Decimate on load rather than inside the model, so discarded frames are
        # never decoded or resized. clip_frames stays the raw window length;
        # every tensor this dataset returns has length T_out.
        self.frame_stride = max(1, int(frame_stride))
        self.T_out = -(-clip_frames // self.frame_stride)  # ceil division

    def __len__(self):
        return len(self.df)

    def _fix_seq_len(self, y):
        y = torch.as_tensor(y, dtype=torch.float32).flatten()
        if y.numel() == self.clip_frames:
            return y
        if y.numel() > self.clip_frames:
            return y[:self.clip_frames]
        return torch.cat([y, y[-1:].repeat(self.clip_frames - y.numel())], dim=0)

    def _load_rr_seq(self, row):
        if "rr_path" in row and pd.notna(row["rr_path"]):
            path = str(row["rr_path"])
            if path.endswith(".npy"):
                arr = np.load(path)
            elif path.endswith(".npz"):
                data = np.load(path)
                arr = data["rr"] if "rr" in data else data[list(data.keys())[0]]
            elif path.endswith(".csv"):
                arr = pd.read_csv(path, header=None).values.squeeze()
            else:
                arr = np.loadtxt(path)
            return self._fix_seq_len(arr)

        if "rr_seq" in row and pd.notna(row["rr_seq"]):
            arr = [float(v) for v in str(row["rr_seq"]).replace(",", ";").split(";") if v.strip()]
            return self._fix_seq_len(arr)

        if "rr_bpm" in row and pd.notna(row["rr_bpm"]):
            return torch.full((self.clip_frames,), float(row["rr_bpm"]), dtype=torch.float32)

        raise ValueError("Provide rr_path, rr_seq, or rr_bpm.")

    def _load_rr_mask(self, row):
        """Per-frame validity mask, 1 = supervise on this frame, 0 = ignore.

        Absent column means every frame is valid, which keeps older metadata
        CSVs working unchanged.
        """
        if "rr_mask_path" in row and pd.notna(row["rr_mask_path"]):
            arr = np.load(str(row["rr_mask_path"]))
            return (self._fix_seq_len(arr) > 0.5).float()
        return torch.ones(self.clip_frames, dtype=torch.float32)

    def _fix_clip_len(self, x):
        T = x.shape[0]
        if T == self.T_out:
            return x
        if T > self.T_out:
            return x[:self.T_out]
        return torch.cat([x, x[-1:].repeat(self.T_out - T, 1, 1, 1)], dim=0)

    def _stride_seq(self, y):
        """Decimate a full-length (clip_frames) label/mask to T_out."""
        if self.frame_stride == 1:
            return y
        y = y[::self.frame_stride][:self.T_out]
        if y.numel() < self.T_out:
            y = torch.cat([y, y[-1:].repeat(self.T_out - y.numel())], dim=0)
        return y

    def _resize(self, x, mode="bilinear"):
        if mode == "nearest":
            return F.interpolate(x, size=(self.input_size, self.input_size), mode=mode)
        return F.interpolate(x, size=(self.input_size, self.input_size), mode=mode, align_corners=False)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip = load_clip(row["clip_path"], stride=self.frame_stride)
        if clip.shape[1] < 3:
            clip = clip.repeat(1, 3, 1, 1)
        clip = clip[:, :3]
        if self.normalize:
            clip = normalize_to_01(clip)
        clip = self._fix_clip_len(clip)
        clip = self._resize(clip)

        # Labels are stored at full frame rate, so decimate them with the same
        # stride to stay aligned with the clip frames actually loaded.
        rr_seq = self._stride_seq(self._load_rr_seq(row))
        rr_mask = self._stride_seq(self._load_rr_mask(row))
        # Clip-level scalar averages only the supervised frames.
        rr_scalar = (rr_seq * rr_mask).sum() / rr_mask.sum().clamp_min(1.0)

        cond = str(row["cond"]).lower() if "cond" in row and pd.notna(row["cond"]) else "exposed"
        cond_id = torch.tensor(COND_TO_ID.get(cond, 0), dtype=torch.long)

        sample = {
            "clip": clip,
            "rr_seq": rr_seq,
            "rr_mask": rr_mask,
            "rr": rr_scalar,
            "cond_id": cond_id,
            "clip_path": str(row["clip_path"]),
            # Spec sections 33/35: the evaluation CSVs are keyed on clip_id.
            # Fall back to the path when older metadata has no such column.
            "clip_id": (str(row["clip_id"]) if "clip_id" in row and pd.notna(row["clip_id"])
                        else str(row["clip_path"])),
        }

        if self.include_temp:
            has_temp = "temp_c" in row and pd.notna(row["temp_c"])
            sample["temp"] = torch.tensor(float(row["temp_c"]) if has_temp else 0.0, dtype=torch.float32)
            sample["temp_valid"] = torch.tensor(1.0 if has_temp else 0.0, dtype=torch.float32)
            mask = None
            if "face_mask_path" in row and pd.notna(row["face_mask_path"]):
                mask = load_mask(row["face_mask_path"], stride=self.frame_stride)
            if mask is None:
                mask = torch.ones((self.T_out, 1, self.input_size, self.input_size), dtype=torch.float32)
            else:
                mask = self._fix_clip_len(mask)
                mask = self._resize(mask, mode="nearest")
            sample["face_mask"] = mask

        return sample
