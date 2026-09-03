import argparse
from pathlib import Path
import torch
import pandas as pd
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from macro_motion.models.unet import UNetNIR2IR
from macro_motion.utils.io import load_clip, normalize_to_01
from macro_motion.losses.losses import unet_nir2ir_loss
from macro_motion.utils.device import resolve_device, assert_module_on_cuda

class NIRIRFrameDataset(torch.utils.data.Dataset):
    """
    CSV columns: nir_path, ir_path.
    Each row can point to clips. Frames are paired by index.
    """
    def __init__(self, csv, input_size=224):
        self.df = pd.read_csv(csv)
        self.input_size = input_size
        self.index = []
        self.cache = {}
        for i, row in self.df.iterrows():
            nir = normalize_to_01(load_clip(row["nir_path"]))[:, :3]
            ir = normalize_to_01(load_clip(row["ir_path"]))
            T = min(nir.shape[0], ir.shape[0])
            self.index.extend([(i, t) for t in range(T)])

    def _clip(self, path):
        if path not in self.cache:
            self.cache[path] = normalize_to_01(load_clip(path))
        return self.cache[path]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row_idx, t = self.index[idx]
        row = self.df.iloc[row_idx]
        nir = self._clip(row["nir_path"])[t:t+1, :3]
        ir = self._clip(row["ir_path"])[t:t+1]
        if ir.shape[1] > 1: ir = ir[:, :1]
        nir = F.interpolate(nir, size=(self.input_size,self.input_size), mode="bilinear", align_corners=False)[0]
        ir = F.interpolate(ir, size=(self.input_size,self.input_size), mode="bilinear", align_corners=False)[0]
        return {"nir": nir, "ir": ir}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--ssim_w", type=float, default=0.1)
    p.add_argument("--allow-cpu", "--allow_cpu", dest="allow_cpu", action="store_true",
                   help="permit running without a GPU (debugging only; very slow)")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(allow_cpu=args.allow_cpu)
    ds = NIRIRFrameDataset(args.pairs_csv, args.input_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = UNetNIR2IR(base=args.base).to(device)
    assert_module_on_cuda(model, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, args.epochs+1):
        losses = []
        model.train()
        for batch in tqdm(loader, desc=f"unet epoch {epoch}"):
            nir, ir = batch["nir"].to(device), batch["ir"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                pred = model(nir)
                loss, _ = unet_nir2ir_loss(pred, ir, mse_w=1.0, ssim_w=args.ssim_w)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu()))
        print({"epoch": epoch, "loss": sum(losses)/max(1,len(losses))})
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last_unet.pt")

if __name__ == "__main__":
    main()
