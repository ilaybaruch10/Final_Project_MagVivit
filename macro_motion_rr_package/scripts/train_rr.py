import argparse, json, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from macro_motion.utils.seed import set_seed
from macro_motion.utils.device import resolve_device, assert_module_on_cuda
from macro_motion.data.datamodule import make_loaders
from macro_motion.data.alignment import check_alignment
from macro_motion.data.dataset import MacroMotionDataset
from macro_motion.models.macro_motion_model import MacroMotionModel
from macro_motion.losses.losses import UncertaintyWeightedRRTempLoss
from macro_motion.metrics.rr_metrics import compute_regression_metrics

def str2bool(x):
    if isinstance(x, bool): return x
    return str(x).lower() in ["1", "true", "yes", "y"]

def resolve_rr_init_bias(cfg, metadata_csv):
    """Config 'auto' -> mean RR over supervised frames of the train split.

    Reads only the small label/mask .npy files, not the clips.
    """
    val = cfg["model"].get("rr_init_bias", 0.0)
    if not (isinstance(val, str) and val.lower() == "auto"):
        return float(val or 0.0)
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == "train"]
    tot = cnt = 0.0
    for _, r in df.iterrows():
        if "rr_path" not in r or pd.isna(r["rr_path"]):
            continue
        y = np.load(str(r["rr_path"])).astype("float64").ravel()
        m = (np.load(str(r["rr_mask_path"])).astype("float64").ravel()
             if "rr_mask_path" in r and pd.notna(r["rr_mask_path"]) else np.ones_like(y))
        n = min(len(y), len(m))
        tot += float((y[:n] * m[:n]).sum()); cnt += float(m[:n].sum())
    bias = tot / cnt if cnt else 0.0
    print(f"[init] rr_init_bias=auto -> {bias:.2f} bpm (mean over {int(cnt)} supervised train frames)")
    return bias

def downsample_rr_seq(rr_seq_raw, temporal_downsample):
    if temporal_downsample is None or temporal_downsample <= 1:
        return rr_seq_raw
    return rr_seq_raw[:, ::temporal_downsample]

@torch.no_grad()
def evaluate(model, loader, device, include_temp, criterion, cfg):
    model.eval()
    ys_frame, ps_frame = [], []
    total_loss, n = 0.0, 0
    ds = cfg["model"]["temporal_downsample"]

    for batch in loader:
        clip = batch["clip"].to(device)
        rr_seq = downsample_rr_seq(batch["rr_seq"].to(device), ds)
        rr_mask = downsample_rr_seq(batch["rr_mask"].to(device), ds)
        cond_id = batch["cond_id"].to(device)
        face_mask = batch.get("face_mask", None)
        if face_mask is not None: face_mask = face_mask.to(device)

        preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
        loss, _ = criterion(preds, rr_seq, rr_mask=rr_mask)

        total_loss += loss.item() * clip.size(0); n += clip.size(0)
        # Score only supervised frames, so masked ones cannot flatter or spoil metrics.
        keep = rr_mask.detach().cpu().reshape(-1).numpy() > 0.5
        ys_frame.extend(rr_seq.detach().cpu().reshape(-1).numpy()[keep].tolist())
        ps_frame.extend(preds["rr_pred_seq"].detach().cpu().reshape(-1).numpy()[keep].tolist())

    metrics = compute_regression_metrics(ys_frame, ps_frame)
    return {**{f"frame_{k}": v for k, v in metrics.items()}, "loss": total_loss / max(n,1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--include_temp_path", default=None)
    p.add_argument("--allow-cpu", "--allow_cpu", dest="allow_cpu", action="store_true",
                   help="permit running without a GPU (debugging only; very slow)")
    p.add_argument("--skip-alignment-check", action="store_true",
                   help="skip the spec section 11 image<->RR alignment check. It decodes "
                        "two clips (a few seconds); skip only when iterating on unrelated code.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.include_temp_path is not None:
        cfg["model"]["include_temp_path"] = str2bool(args.include_temp_path)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    set_seed(cfg["seed"])
    device = resolve_device(allow_cpu=args.allow_cpu,
                            require_cuda=bool(cfg["train"].get("require_cuda", True)))
    include_temp = bool(cfg["model"]["include_temp_path"])

    frame_stride = int(cfg["data"].get("frame_stride", 1))
    if frame_stride > 1 and int(cfg["model"]["temporal_downsample"]) > 1:
        raise ValueError(
            f"data.frame_stride={frame_stride} and model.temporal_downsample="
            f"{cfg['model']['temporal_downsample']} would decimate twice "
            f"(effective stride {frame_stride * int(cfg['model']['temporal_downsample'])}). "
            f"Set model.temporal_downsample: 1 when decimating in the dataset."
        )
    print(f"[data] frame_stride={frame_stride} -> {-(-int(cfg['data']['clip_frames']) // frame_stride)} "
          f"frames/clip from {cfg['data']['clip_frames']} raw")

    # Spec section 11: verify image <-> RR alignment BEFORE training begins. A
    # silent misalignment still trains, still lowers the loss, and produces a
    # result that means nothing, so this runs unconditionally rather than behind
    # a debug flag.
    if not args.skip_alignment_check:
        check_alignment(
            MacroMotionDataset(args.metadata, "train", cfg["data"]["input_size"],
                               cfg["data"]["clip_frames"], True, include_temp,
                               frame_stride),
            metadata_csv=args.metadata, n_clips=2)

    train_loader, val_loader, _ = make_loaders(
        args.metadata, cfg["data"]["input_size"], cfg["data"]["clip_frames"],
        cfg["train"]["batch_size"], cfg["train"]["num_workers"], include_temp,
        frame_stride=frame_stride
    )

    model = MacroMotionModel(
        d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
        depth=cfg["model"]["depth"], dropout=cfg["model"]["dropout"],
        ff_mult=cfg["model"]["ff_mult"], max_len=cfg["model"]["max_len"],
        temporal_downsample=cfg["model"]["temporal_downsample"],
        pretrained_resnet=cfg["model"]["use_pretrained_resnet"],
        freeze_resnet_until=cfg["model"]["freeze_resnet_until"],
        include_temp_path=include_temp, unet_base=cfg["model"]["unet_base"],
        use_domain_token=cfg["model"].get("use_domain_token", False),
        rr_init_bias=resolve_rr_init_bias(cfg, args.metadata)
    ).to(device)
    assert_module_on_cuda(model, device)

    # Default to uncertainty weighting only when there are two heads to balance.
    use_unc = bool(cfg["train"].get("use_uncertainty", include_temp))
    criterion = UncertaintyWeightedRRTempLoss(
        cfg["train"]["rr_loss"],
        cfg["train"]["huber_delta"],
        cfg["train"].get("rr_smooth_w", 0.05),
        use_uncertainty=use_unc
    ).to(device)
    print(f"[loss] {cfg['train']['rr_loss']} delta={cfg['train']['huber_delta']} "
          f"smooth_w={cfg['train'].get('rr_smooth_w', 0.05)} uncertainty={use_unc}")

    # criterion.parameters() is empty unless uncertainty weighting is active
    # (spec section 28), so this adds nothing in the RR-only configuration.
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))
    best_mae = float("inf")
    ds = cfg["model"]["temporal_downsample"]

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        # Per-epoch train accumulators. Without these only val was recorded, so a
        # stuck model was indistinguishable from a model fitting while val lagged.
        tr_loss = tr_rr = tr_n = 0.0
        tr_gnorm = 0.0
        pred_sum = pred_sqsum = pred_cnt = 0.0
        for batch in pbar:
            clip = batch["clip"].to(device)
            rr_seq = downsample_rr_seq(batch["rr_seq"].to(device), ds)
            rr_mask = downsample_rr_seq(batch["rr_mask"].to(device), ds)
            cond_id = batch["cond_id"].to(device)
            face_mask = batch.get("face_mask", None)
            if face_mask is not None: face_mask = face_mask.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg["train"]["amp"])):
                preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
                loss, terms = criterion(preds, rr_seq, rr_mask=rr_mask)

            scaler.scale(loss).backward()
            gnorm = float("nan")
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                # clip_grad_norm_ returns the pre-clip global norm.
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"]))
            scaler.step(opt); scaler.update()

            bs = clip.size(0)
            tr_loss += float(loss.detach().cpu()) * bs
            tr_rr += float(terms["rr_loss"].cpu()) * bs
            tr_gnorm += (gnorm if gnorm == gnorm else 0.0) * bs
            tr_n += bs
            # Spread of the predictions themselves: a near-zero std means the head
            # is emitting a constant, which is the failure mode worth catching.
            p = preds["rr_pred_seq"].detach().float().cpu()
            m = rr_mask.detach().float().cpu()
            pred_sum += float((p * m).sum()); pred_sqsum += float(((p ** 2) * m).sum())
            pred_cnt += float(m.sum())
            pbar.set_postfix(loss=float(loss.detach().cpu()), rr=float(terms["rr_loss"].cpu()))

        n = max(tr_n, 1e-9)
        pm = pred_sum / max(pred_cnt, 1e-9)
        pstd = max(pred_sqsum / max(pred_cnt, 1e-9) - pm * pm, 0.0) ** 0.5
        train_rec = {
            "train_loss": tr_loss / n,
            "train_rr_loss": tr_rr / n,
            "train_grad_norm": tr_gnorm / n,
            "train_pred_mean": pm,
            "train_pred_std": pstd,
            # None whenever uncertainty weighting is off (spec section 28).
            "logsig_rr": (float(criterion.logsig_rr.detach().cpu())
                          if criterion.logsig_rr is not None else None),
            "steps_per_epoch": len(train_loader),
        }

        val = evaluate(model, val_loader, device, include_temp, criterion, cfg)
        print(f"train_loss={train_rec['train_loss']:.4f} rr={train_rec['train_rr_loss']:.4f} "
              f"pred_mean={pm:.2f} pred_std={pstd:.3f} gnorm={train_rec['train_grad_norm']:.3f}")
        print("val", val)
        with open(out_dir / "val_log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **train_rec, **val}) + "\n")
        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(), "cfg": cfg, "epoch": epoch, "val_metrics": val}
        if val["frame_MAE"] < best_mae:
            best_mae = val["frame_MAE"]
            torch.save(ckpt, out_dir / "best.pt")
        if epoch % cfg["train"]["save_every"] == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")
    print("Best val frame MAE:", best_mae)

if __name__ == "__main__":
    main()
