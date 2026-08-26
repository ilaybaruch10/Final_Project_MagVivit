import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssim import ssim_1ch

class UncertaintyWeightedRRTempLoss(nn.Module):
    """Kendall-style uncertainty weighting over the RR and Temp heads.

    `use_uncertainty=False` disables the weighting and optimises the raw task
    loss. Use that whenever only one head is active: with a single task there is
    nothing to balance, and `0.5*exp(-s)*L + 0.5*s` is then just an escape hatch
    -- the optimiser can lower the reported total by raising `s` instead of by
    improving the prediction. In the 60-epoch overfit probe it did exactly that:
    over the run the true RR loss rose 320.6 -> 364.4 while the logged total fell
    160.3 -> 144.8, bought entirely by `logsig_rr` drifting 0 -> 0.23.
    """

    def __init__(self, rr_loss="huber", huber_delta=2.0, rr_smooth_w=0.05,
                 use_uncertainty=True):
        super().__init__()
        self.rr_loss = rr_loss
        self.huber_delta = huber_delta
        self.rr_smooth_w = rr_smooth_w
        self.use_uncertainty = bool(use_uncertainty)
        # Spec section 28: with RR the only active task there must be NO logsig_rr.
        # Registering it unconditionally left a learnable scalar in the optimiser
        # that weight decay still pulled on, and logged a number that looked
        # meaningful while taking no part in the loss. Create them only when the
        # weighting is genuinely active.
        if self.use_uncertainty:
            self.logsig_rr = nn.Parameter(torch.tensor(0.0))
            self.logsig_temp = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("logsig_rr", None)
            self.register_parameter("logsig_temp", None)

    def base_rr_loss(self, pred, target):
        if self.rr_loss == "huber":
            return F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")
        if self.rr_loss == "mse":
            return F.mse_loss(pred, target, reduction="none")
        if self.rr_loss == "l1":
            return F.l1_loss(pred, target, reduction="none")
        raise ValueError(self.rr_loss)

    @staticmethod
    def smoothness_loss(seq, mask=None):
        """Mean |RR_t - RR_{t-1}|, counting a step only when both frames are valid.

        Across a masked gap the two surviving neighbours are not temporally
        adjacent, so penalising the jump between them would be meaningless.
        """
        if seq.ndim != 2 or seq.shape[1] < 2:
            return torch.tensor(0.0, device=seq.device)
        step = (seq[:, 1:] - seq[:, :-1]).abs()
        if mask is None:
            return step.mean()
        pair = mask[:, 1:] * mask[:, :-1]
        return (step * pair).sum() / pair.sum().clamp_min(1.0)

    def forward(self, preds, rr, temp=None, temp_valid=None, w_temp=None, rr_mask=None):
        if "rr_pred_seq" in preds and rr.ndim == 2:
            pred_rr = preds["rr_pred_seq"]
            if pred_rr.shape != rr.shape:
                raise ValueError(f"RR sequence shape mismatch: pred={pred_rr.shape}, target={rr.shape}")
            if rr_mask is not None and rr_mask.shape != rr.shape:
                raise ValueError(f"RR mask shape mismatch: mask={rr_mask.shape}, target={rr.shape}")
            rr_loss_vec = self.base_rr_loss(pred_rr, rr)  # (B,T)
            if rr_mask is None:
                rr_base = rr_loss_vec.mean(dim=1)         # (B,)
            else:
                # Per-clip mean over supervised frames only. A clip with no
                # valid frame contributes 0 rather than NaN.
                rr_base = (rr_loss_vec * rr_mask).sum(dim=1) / rr_mask.sum(dim=1).clamp_min(1.0)
            rr_smooth = self.smoothness_loss(pred_rr, rr_mask)
        else:
            pred_rr = preds["rr_pred"]
            rr_loss_vec = self.base_rr_loss(pred_rr, rr.view_as(pred_rr))
            rr_base = rr_loss_vec
            rr_smooth = torch.tensor(0.0, device=pred_rr.device)

        if self.use_uncertainty:
            rr_unc = 0.5 * torch.exp(-self.logsig_rr) * rr_base + 0.5 * self.logsig_rr
        else:
            rr_unc = rr_base
        total = rr_unc.mean() + self.rr_smooth_w * rr_smooth
        # logsig_rr is reported so a falling total can always be attributed:
        # either the prediction improved or the weighting was relaxed.
        terms = {"rr_loss": rr_base.mean().detach(), "rr_unc": rr_unc.mean().detach(),
                 "rr_smooth": rr_smooth.detach(),
                 "logsig_rr": (self.logsig_rr.detach() if self.logsig_rr is not None
                               else torch.zeros((), device=pred_rr.device))}

        if "temp_pred" in preds and temp is not None:
            temp_loss_vec = F.mse_loss(preds["temp_pred"], temp, reduction="none")
            if temp_valid is None:
                temp_valid = torch.ones_like(temp_loss_vec)
            if w_temp is None:
                w_temp = torch.ones_like(temp_loss_vec)
            if self.use_uncertainty:
                temp_unc = 0.5 * torch.exp(-self.logsig_temp) * temp_loss_vec + 0.5 * self.logsig_temp
            else:
                temp_unc = temp_loss_vec
            temp_unc = temp_unc * temp_valid * w_temp
            total = total + temp_unc.mean()
            terms["temp_loss"] = temp_loss_vec.mean().detach()
            terms["temp_unc"] = temp_unc.mean().detach()

        return total, terms

def unet_nir2ir_loss(ir_hat, ir_gt, mse_w=1.0, ssim_w=0.1):
    mse = F.mse_loss(ir_hat, ir_gt)
    ssim_loss = 1.0 - ssim_1ch(ir_hat, ir_gt)
    return mse_w*mse + ssim_w*ssim_loss, {"mse": mse.detach(), "ssim": (1.0-ssim_loss).detach()}
