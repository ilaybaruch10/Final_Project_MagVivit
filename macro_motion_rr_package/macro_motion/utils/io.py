from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load_clip(path, stride=1):
    """
    Returns a tensor with shape (T,C,H,W).
    Supports .npy, .npz with key 'frames', or folder of images.

    ``stride`` keeps only every Nth frame. For a folder it is applied to the
    sorted file list *before* decoding, so frames that would be discarded
    downstream are never read off disk or resized. PNG decode dominates the cost
    of this pipeline, so stride=3 is roughly a 3x saving in I/O and peak memory.
    """
    stride = max(1, int(stride))
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)[::stride]
    elif path.suffix == ".npz":
        data = np.load(path)
        arr = data["frames"] if "frames" in data else data[list(data.keys())[0]]
        arr = arr[::stride]
    elif path.is_dir():
        frame_paths = [p for p in sorted(path.iterdir())
                       if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]]
        if not frame_paths:
            raise ValueError(f"No image frames found in {path}")
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths[::stride]]
        arr = np.stack(imgs, axis=0)
    else:
        raise ValueError(f"Unsupported clip path: {path}")

    if arr.ndim != 4:
        raise ValueError(f"Expected 4D clip, got {arr.shape}")

    if arr.shape[-1] in [1, 2, 3, 4]:
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.shape[1] in [1, 2, 3, 4]:
        pass
    else:
        raise ValueError(f"Cannot infer channel dimension for shape {arr.shape}")

    if arr.shape[1] == 4:
        arr = arr[:, :3]

    return torch.from_numpy(arr)

def normalize_to_01(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            maxv = float(torch.iinfo(x.dtype).max)
            x = x.float() / maxv
    else:
        x = x.float()
        if x.numel() > 0 and x.max() > 2.0:
            denom = 65535.0 if x.max() > 255.0 else 255.0
            x = x / denom
    return x.clamp(0.0, 1.0)

def load_mask(path, stride=1):
    if path is None or str(path) == "" or str(path).lower() == "nan":
        return None
    m = load_clip(path, stride=stride)
    if m.shape[1] > 1:
        m = m[:, :1]
    m = normalize_to_01(m)
    return (m > 0.5).float()
