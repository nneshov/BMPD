import torch
from torch import nn
from torchvision import models

MODES = ("bmpd", "framediff")


def frame_difference(x):
    if x.ndim != 4 or tuple(x.shape[1:]) != (20, 50, 56):
        raise ValueError("Expected [B,20,50,56], got %r" % (tuple(x.shape),))
    return x[:, 1:] - x[:, :-1]


def padded_control(x):
    d = frame_difference(x)
    return torch.cat((d, torch.zeros_like(d[:, :1])), dim=1)


def reference_scale(rms, eps=1e-6):
    flat = rms.flatten(1)
    median = flat.median(dim=1).values[:, None, None, None]
    mean = flat.mean(dim=1)[:, None, None, None]
    return torch.where(median > eps, median, mean).clamp_min(eps)


def bmpd(x, tau=0.10, eps=1e-6):
    d = frame_difference(x)
    r = d.square().mean(dim=1, keepdim=True).sqrt()
    s = reference_scale(r, eps)
    denominator = (r.square() + (tau * s).square()).sqrt().clamp_min(eps)
    q = d / denominator
    a = torch.log1p(r / s)
    result = torch.cat((q, a), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite BMPD representation")
    return result


class RepresentationModel(nn.Module):

    def __init__(self, mode, tau, epsilon):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown mode: %s" % mode)
        self.mode = mode
        self.tau = float(tau)
        self.epsilon = float(epsilon)

    def transform(self, x):
        if self.mode == "framediff":
            return padded_control(x)
        return bmpd(x, self.tau, self.epsilon)


class LiteCNN(RepresentationModel):

    def __init__(self, mode, tau=0.10, epsilon=1e-6, output_dim=8):
        super().__init__(mode, tau, epsilon)
        self.features = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21504, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, return_embedding=False):
        h = self.features(self.transform(x))
        if tuple(h.shape[1:]) != (128, 12, 14):
            raise AssertionError("Unexpected feature shape: %r" % (tuple(h.shape),))
        h = self.classifier[0](h)
        embedding = self.classifier[2](self.classifier[1](h))
        logits = self.classifier[4](self.classifier[3](embedding))
        return (logits, embedding) if return_embedding else logits


class ResNet18(RepresentationModel):

    def __init__(self, mode, tau=0.10, epsilon=1e-6, output_dim=8):
        super().__init__(mode, tau, epsilon)
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            20, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x, return_embedding=False):
        if return_embedding:
            raise ValueError("ResNet-18 does not expose the LiteCNN embedding")
        return self.model(self.transform(x))


def build_model(backbone, mode, tau=0.10, epsilon=1e-6):
    if backbone == "litecnn":
        return LiteCNN(mode, tau, epsilon)
    if backbone == "resnet18":
        return ResNet18(mode, tau, epsilon)
    raise ValueError("Unknown backbone: %s" % backbone)
