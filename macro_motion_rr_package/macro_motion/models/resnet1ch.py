import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def make_frozen_resnet50_feature_extractor(pretrained=True):
    """
    Built-in PyTorch ResNet-50, unchanged conv1=3 channels.
    Frozen feature extractor:
      input:  (N,3,H,W)
      output: (N,2048)
    When pretrained=False, weights are random (no download) — useful for
    offline smoke tests; real training should keep pretrained=True.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = resnet50(weights=weights)
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

def make_frozen_resnet50_trunk(pretrained=True):
    """
    Built-in PyTorch ResNet-50 trunk up to layer4.
    Unchanged conv1=3 channels, frozen.
      input:  (N,3,H,W)
      output: (N,2048,16,16) for 512x512 input
    pretrained=False skips the weight download (random init).
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = resnet50(weights=weights)
    trunk = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3, m.layer4
    )
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad = False
    return trunk
