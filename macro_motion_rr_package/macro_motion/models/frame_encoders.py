import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet1ch import make_frozen_resnet50_feature_extractor, make_frozen_resnet50_trunk

# Spec section 6: normalization expected by the pretrained torchvision ResNet-50
# weights. The dataset delivers frames in [0,1]; the IMAGENET1K_V2 weights were
# trained on inputs standardised with these per-channel constants, and the frozen
# BatchNorm running statistics inside the trunk encode that same input
# distribution. Feeding raw [0,1] leaves every channel roughly +2 sigma off the
# mean the BN layers expect, so the pretrained features are evaluated far outside
# their calibrated range. No clipping is applied afterwards -- the output is not
# restricted to [0,1] and is not supposed to be.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNormalize(nn.Module):
    """(x - mean) / std over the channel axis, for input already scaled to [0,1].

    Registered as buffers so the constants follow the module across .to(device)
    and are captured in the checkpoint, making the preprocessing an explicit,
    inspectable part of the model rather than a convention callers must remember.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):                     # x: (N,3,H,W) in [0,1]
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)


class FrozenResNet50FrameEncoder(nn.Module):
    """
    Frozen built-in PyTorch pretrained ResNet-50 feature extractor.

    Input:
      x: (B,T,3,H,W) for the RR path.

    The built-in ResNet-50 is NOT modified and NOT fine-tuned.
    A fallback for 1-channel input still exists, but the RR path uses 3-channel input directly.

    Output:
      z: (B,T,d_model)
    """
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        self.backbone = make_frozen_resnet50_feature_extractor(pretrained=pretrained)
        self.normalize = ImageNetNormalize()      # spec section 6
        self.proj = nn.Linear(2048, d_model)

    def train(self, mode: bool = True):
        """Keep the frozen trunk in eval mode even when the model is set to train.

        nn.Module.train() recurses into every child, so a plain `model.train()` in
        the training loop flips the ResNet's BatchNorm layers back to batch
        statistics. forward() re-forces eval before the trunk runs, so the numbers
        were never wrong -- but that leaves a window where the module reports a
        state it does not have, and it only holds for callers who go through
        forward(). Anything touching `enc.backbone` directly, or inspecting
        `.training` to decide what to do, would be misled. Overriding train() makes
        the invariant true at all times rather than repaired just in time.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def to_3ch(x):
        # RR path should already be 3-channel and passes through unchanged.
        if x.shape[2] == 3:
            return x
        # Fallback only for optional modules / experiments.
        if x.shape[2] == 1:
            return x.repeat(1, 1, 3, 1, 1)
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[2]}")

    def forward(self, x):
        B,T,C,H,W = x.shape
        # Spec section 14: merge batch and time so one shared ResNet sees all frames.
        x = self.to_3ch(x).reshape(B*T, 3, H, W)
        x = self.normalize(x)                     # spec section 6, [0,1] -> ImageNet scale

        # Spec section 13: the trunk must stay in eval mode even under model.train(),
        # or BatchNorm renormalises by the current clip and destroys the between-clip
        # signal. MacroMotionModel.train() also enforces this; asserting here keeps the
        # encoder correct when used standalone.
        self.backbone.eval()
        # ResNet is frozen; no gradients/activation storage through it.
        with torch.no_grad():
            feat = self.backbone(x)  # (B*T,2048)

        z = self.proj(feat).view(B, T, -1)
        return z


class FrozenResNet50SpatialBackbone(nn.Module):
    """
    Frozen built-in PyTorch pretrained ResNet-50 trunk up to layer4.

    Input:
      x: (N,1,H,W) or (N,3,H,W)

    If 1-channel, repeat to 3 channels.
    Output:
      fmap: (N,2048,7,7)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.trunk = make_frozen_resnet50_trunk(pretrained=pretrained)
        self.out_channels = 2048

    @staticmethod
    def to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 3:
            return x
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

    def forward(self, x):
        x = self.to_3ch(x)
        with torch.no_grad():
            return self.trunk(x)


class FaceAwareFrozenResNet50Encoder(nn.Module):
    """
    Temp path encoder with frozen built-in pretrained ResNet-50 trunk
    and trainable face-aware attention pooling + projection.

    Input:
      frames: (B,T,1,H,W) or (B,T,3,H,W)
      face_mask: (B,T,1,h,w), optional. Resized to the ResNet feature-map grid
                 (H//32) if it doesn't already match.

    Output:
      z: (B,T,d_model)
    """
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        self.backbone = FrozenResNet50SpatialBackbone(pretrained=pretrained)
        self.attn = nn.Conv2d(2048, 1, 1)
        self.softplus = nn.Softplus()
        self.proj = nn.Linear(2048, d_model)

    def forward(self, frames, face_mask=None):
        B,T,C,H,W = frames.shape
        fmap = self.backbone(frames.reshape(B*T, C, H, W))  # (B*T,2048,h,w), h=w=H//32

        A = self.softplus(self.attn(fmap)) + 1e-6           # (B*T,1,h,w)

        if face_mask is None:
            M = torch.ones_like(A)
        else:
            M = face_mask.reshape(B*T, 1, *face_mask.shape[-2:]).to(A.device).clamp(0, 1)
            if M.shape[-2:] != A.shape[-2:]:                # defensive: match feature-map grid
                M = F.interpolate(M, size=A.shape[-2:], mode="nearest")

        Wm = A * M
        Wm = Wm / Wm.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)

        pooled = (fmap * Wm).sum(dim=(2,3))                 # (B*T,2048)
        return self.proj(pooled).view(B, T, -1)
