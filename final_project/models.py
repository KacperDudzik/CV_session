"""Model builders: one function for every architecture from class.

simple_cnn       Session 7 SimpleCNN (from scratch)
resnet18         Session 7 ImageNet ResNet-18, new fc
vit_b_16         Session 9 ImageNet ViT-B/16, new head on the [CLS] token
vgg16            Session 7 slide 18 (3x3 convs only; dropout 0.5 inside its classifier)
densenet121      Session 7 homework (dense connections instead of residual additions)
efficientnet_b0  Session 7 homework (compound-scaled CNN; dropout 0.2 before its head)

An experiment spec (config.EXPERIMENTS) decides:
  train    parameter-name prefixes that are trained (Session 7 rule); None = everything
  dropout  nn.Dropout(p) before the new last layer (Session 6, slide 32); 0 = none
  norm     "imagenet" (pretrained networks must use it, S7 slide 24) or "train"
           (training-set mean/std, S3; only for networks trained from scratch)
"""

import torch
import torch.nn as nn
from torchvision import models as tvm

import session7 as s7  # SimpleCNN, make_partially_unfrozen_resnet, prep, device


def _head(d_in: int, n_classes: int, dropout: float) -> nn.Module:
    linear = nn.Linear(d_in, n_classes)
    return nn.Sequential(nn.Dropout(dropout), linear) if dropout else linear


class TrainNorm(nn.Module):
    """Input arrives ImageNet-normalised (session7.prep); re-normalise with training stats."""

    def __init__(self, net: nn.Module, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x * self.im_std + self.im_mean - self.mean) / self.std)


def build_model(spec: dict, n_classes: int, seed: int = 0, norm_stats=None):
    """Return (model, trainable parameters) for one experiment spec."""
    arch, train, p = spec["arch"], spec["train"], spec.get("dropout", 0.0)
    torch.manual_seed(seed)
    if arch == "resnet18":
        model, _ = s7.make_partially_unfrozen_resnet(train, n_classes=n_classes, seed=seed)
        if p:
            model.fc = _head(512, n_classes, p)
    elif arch == "simple_cnn":
        model = s7.SimpleCNN(n_classes)
        if p:
            model.classifier = nn.Sequential(nn.Flatten(), _head(32 * 56 * 56, n_classes, p))
    elif arch == "vit_b_16":
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = _head(768, n_classes, p)
    elif arch == "densenet121":
        model = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = _head(1024, n_classes, p)
    elif arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, n_classes)      # classifier[0] = its Dropout(0.2)
        if p:
            model.classifier[0].p = p
    elif arch == "vgg16":
        model = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, n_classes)      # classifier keeps Dropout(0.5) x2
        if p:
            model.classifier[2].p = model.classifier[5].p = p
    else:
        raise ValueError(f"unknown arch {arch!r}")

    for name, param in model.named_parameters():
        param.requires_grad = train is None or any(name.startswith(prefix) for prefix in train)
    if spec.get("norm", "imagenet") == "train":
        assert train is None, "train-set normalisation is only for networks trained from scratch"
        mean, std = norm_stats or ([0.5] * 3, [0.25] * 3)  # placeholder; real values come from the checkpoint
        model = TrainNorm(model, mean, std)
    return model, [q for q in model.parameters() if q.requires_grad]


def load_checkpoint(path):
    """Rebuild a saved model from its checkpoint; returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ckpt.get("spec") or {"arch": ckpt["arch"], "train": ckpt["train"]}
    model, _ = build_model(spec, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(s7.device).eval(), ckpt


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Softmax probabilities (n, n_classes) for a uint8 image tensor; Session 7 prep()."""
    model.eval()
    chunks = [
        torch.softmax(model(s7.prep(X[i:i + batch_size])), dim=1).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(chunks)
