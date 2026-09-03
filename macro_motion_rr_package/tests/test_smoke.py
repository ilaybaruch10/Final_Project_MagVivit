"""
Smoke tests for the macro_motion RR pipeline.

These are NOT accuracy tests — they only assert that the data path, model
forward, training loop, and evaluation run end-to-end without error and
produce the expected artifacts. They use random-init ResNet
(use_pretrained_resnet=false) so they need no network, and tiny shapes so
they finish in seconds on CPU.

Run:  pytest -q   (from inside macro_motion_rr_package/)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_model_forward_runs():
    """MacroMotionModel forward produces the documented RR output shapes."""
    from macro_motion.models.macro_motion_model import MacroMotionModel

    B, T, H, W = 2, 24, 32, 32
    ds = 3
    model = MacroMotionModel(
        d_model=64, n_heads=4, depth=1, ff_mult=2, max_len=32,
        temporal_downsample=ds, pretrained_resnet=False,  # no weight download
        include_temp_path=False,
    ).eval()

    clip = torch.rand(B, T, 3, H, W)
    cond_id = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out = model(clip, cond_id)

    t_model = len(range(0, T, ds))
    assert out["rr_pred_seq"].shape == (B, t_model)
    assert out["rr_pred"].shape == (B,)
    assert torch.isfinite(out["rr_pred"]).all()


def test_dataset_sample_shapes(tmp_path):
    """MacroMotionDataset yields a clip tensor and RR labels from a CSV."""
    import pandas as pd
    from macro_motion.data.dataset import MacroMotionDataset

    clip = (np.random.rand(20, 32, 32, 3) * 255).astype(np.uint8)
    clip_path = tmp_path / "clip0.npy"
    np.save(clip_path, clip)
    csv = tmp_path / "meta.csv"
    pd.DataFrame([{"clip_path": str(clip_path), "rr_bpm": 42.0,
                   "subject_id": "s0", "split": "train"}]).to_csv(csv, index=False)

    ds = MacroMotionDataset(str(csv), split="train", input_size=32, clip_frames=30)
    sample = ds[0]
    assert sample["clip"].shape == (30, 3, 32, 32)        # padded to clip_frames
    assert sample["rr_seq"].shape == (30,)
    assert abs(float(sample["rr"]) - 42.0) < 1e-4          # scalar broadcast


def _write_tiny_config(path):
    cfg = {
        "seed": 42,
        "data": {"input_size": 32, "clip_frames": 24, "raw_fps": 30.0, "normalize": True},
        "model": {
            "d_model": 64, "n_heads": 4, "depth": 1, "dropout": 0.0, "ff_mult": 2,
            "max_len": 32, "temporal_downsample": 3, "use_pretrained_resnet": False,
            "freeze_resnet_until": "", "include_temp_path": False,
            "use_domain_token": False, "unet_base": 16,
        },
        "train": {
            "epochs": 1, "batch_size": 2, "lr": 0.0002, "weight_decay": 0.01,
            "num_workers": 0, "amp": False, "grad_clip": 1.0, "rr_loss": "huber",
            "huber_delta": 2.0, "rr_smooth_w": 0.05,
            "temp_loss_weight_blanket": 0.3, "temp_loss_weight_exposed": 1.0,
            "save_every": 1,
        },
    }
    Path(path).write_text(yaml.safe_dump(cfg))


def _run(args, cwd):
    """Run a package script with the repo on PYTHONPATH; fail loudly on error."""
    env = {"PYTHONPATH": str(PACKAGE_ROOT)}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"{args}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc


def test_end_to_end_scripts(tmp_path):
    """generate -> split -> train -> evaluate via the actual CLI scripts."""
    scripts = PACKAGE_ROOT / "scripts"
    data_dir = tmp_path / "data"
    meta = data_dir / "metadata.csv"
    meta_split = data_dir / "metadata_split.csv"
    cfg = tmp_path / "tiny.yaml"
    out_dir = tmp_path / "run"
    eval_dir = out_dir / "eval_test"
    _write_tiny_config(cfg)

    # 8 subjects so the grouped split has enough groups for train/val/test.
    _run([str(scripts / "make_synthetic_data.py"),
          "--out_dir", str(data_dir / "synthetic"), "--metadata", str(meta),
          "--subjects", "8", "--clips_per_subject", "1",
          "--frames", "24", "--size", "32"], cwd=PACKAGE_ROOT)
    assert meta.exists()

    _run([str(scripts / "split_dataset.py"),
          "--metadata", str(meta), "--out_csv", str(meta_split),
          "--group_col", "subject_id"], cwd=PACKAGE_ROOT)
    assert meta_split.exists()

    _run([str(scripts / "train_rr.py"),
          "--metadata", str(meta_split), "--config", str(cfg),
          "--out_dir", str(out_dir), "--include_temp_path", "false"], cwd=PACKAGE_ROOT)
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "val_log.jsonl").exists()

    _run([str(scripts / "evaluate_rr.py"),
          "--metadata", str(meta_split), "--ckpt", str(out_dir / "best.pt"),
          "--split", "test", "--out_dir", str(eval_dir),
          "--num_workers", "0"], cwd=PACKAGE_ROOT)

    metrics_file = eval_dir / "metrics_test.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text())
    assert "frame_MAE" in metrics and np.isfinite(metrics["frame_MAE"])
    assert (eval_dir / "true_vs_pred_framewise.png").exists()
