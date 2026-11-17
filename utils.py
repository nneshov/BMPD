import random
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def atomic_json(path, value):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(value, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    torch.save(value, str(tmp))
    tmp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_torch(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


@torch.no_grad()
def evaluate(model, loader, device, keep=False):
    model.eval()
    cm = torch.zeros(8, 8, dtype=torch.int64)
    preds = []
    labels = []
    loss_sum = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        yd = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, yd, reduction="sum").item())
        p = logits.argmax(1).cpu()
        y = y.long()
        cm += torch.bincount(y * 8 + p, minlength=64).reshape(8, 8)
        n += len(y)
        if keep:
            preds.append(p)
            labels.append(y)
    recall = []
    f1 = []
    for c in range(8):
        tp = float(cm[c, c])
        actual = float(cm[c].sum())
        predicted = float(cm[:, c].sum())
        recall.append(100 * tp / max(1.0, actual))
        f1.append(200 * tp / max(1.0, actual + predicted))
    metrics = {
        "top1": 100 * float(cm.diag().sum()) / n,
        "loss": loss_sum / n,
        "macro_f1": sum(f1) / 8,
        "recalls": recall,
    }
    artifacts = (
        {"confusion": cm, "predictions": torch.cat(preds), "labels": torch.cat(labels)}
        if keep
        else None
    )
    return (metrics, artifacts)


def model_state(model):
    return {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
