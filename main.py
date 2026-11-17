import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data import PPBPDataset, SevenUserClassBatchSampler, loso_split, prepare_data
from model import build_model, bmpd, padded_control, reference_scale
from utils import (
    atomic_json,
    atomic_torch_save,
    evaluate,
    load_torch,
    model_state,
    set_seed,
    sha256,
)

PARAMETERS = {"litecnn": 5605928, "resnet18": 11233928}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_dir, backbones, methods, seeds, users):
    rows = []
    for backbone in backbones:
        for method in methods:
            for seed in seeds:
                for user in users:
                    path = (
                        output_dir
                        / backbone
                        / method
                        / f"seed{seed}"
                        / f"user{user}"
                        / "result.json"
                    )
                    if path.exists():
                        rows.append(json.loads(path.read_text()))
    if not rows:
        return
    write_csv(output_dir / "per_run.csv", rows)
    by_seed, by_user, summary = [], [], []
    for backbone in backbones:
        for method in methods:
            selected = [
                r for r in rows if r["backbone"] == backbone and r["method"] == method
            ]
            for seed in seeds:
                group = [r for r in selected if r["seed"] == seed]
                if len(group) == len(users):
                    by_seed.append(
                        {
                            "backbone": backbone,
                            "method": method,
                            "seed": seed,
                            "users": len(users),
                            "accuracy": float(np.mean([r["accuracy"] for r in group])),
                            "macro_f1": float(np.mean([r["macro_f1"] for r in group])),
                        }
                    )
            for user in users:
                group = [r for r in selected if r["user"] == user]
                if len(group) == len(seeds):
                    values = [r["accuracy"] for r in group]
                    by_user.append(
                        {
                            "backbone": backbone,
                            "method": method,
                            "user": user,
                            "accuracy": float(np.mean(values)),
                            "seed_sd": (
                                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                            ),
                            "macro_f1": float(np.mean([r["macro_f1"] for r in group])),
                        }
                    )
            complete = [
                r
                for r in by_user
                if r["backbone"] == backbone and r["method"] == method
            ]
            if len(complete) == len(users):
                values = [
                    r["accuracy"]
                    for r in by_seed
                    if r["backbone"] == backbone and r["method"] == method
                ]
                summary.append(
                    {
                        "backbone": backbone,
                        "method": method,
                        "users": len(users),
                        "seeds": len(seeds),
                        "accuracy": float(np.mean([r["accuracy"] for r in complete])),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in complete])),
                        "worst_user": min(r["accuracy"] for r in complete),
                        "seed_sd": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        ),
                    }
                )
    for name, values in (
        ("per_seed.csv", by_seed),
        ("per_user.csv", by_user),
        ("summary.csv", summary),
    ):
        if values:
            write_csv(output_dir / name, values)
    expected = len(backbones) * len(methods) * len(seeds) * len(users)
    print(f"Completed {len(rows)}/{expected} runs", flush=True)
    for row in summary:
        print(
            f"{row['backbone']} {row['method']}: accuracy {row['accuracy']:.4f}%, "
            f"macro-F1 {row['macro_f1']:.4f}%",
            flush=True,
        )


def train(model, train_loader, val_loader, sampler, folder, spec, seed, run_id, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    latest, best_path = folder / "latest.pth", folder / "best.pth"
    start_epoch, best_top1, best_epoch, stale, history = 1, -1.0, 0, 0, []
    best_weights = None
    if latest.exists():
        saved = load_torch(latest)
        if saved["run_id"] != run_id:
            raise ValueError("Resume configuration differs. Use a new output directory.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        best_top1, best_epoch = saved["best_top1"], saved["best_epoch"]
        stale = saved.get("stale", 0)
        best_weights, history = saved["best_weights"], saved["history"]
    for epoch in range(start_epoch, spec["epochs"] + 1):
        set_seed(seed + epoch * 100003)
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
        metrics, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_loader.dataset),
                "val_top1": metrics["top1"],
                "val_loss": metrics["loss"],
            }
        )
        print(
            f"Epoch {epoch}/{spec['epochs']}, validation accuracy {metrics['top1']:.2f}%",
            flush=True,
        )
        if metrics["top1"] > best_top1:
            best_top1, best_epoch = metrics["top1"], epoch
            stale = 0
            best_weights = model_state(model)
            atomic_torch_save(
                {
                    "model_state": best_weights,
                    "best_epoch": best_epoch,
                    "val_top1": best_top1,
                    "run_id": run_id,
                    "metadata": {
                        "backbone": run_id[1],
                        "method": run_id[2],
                        "seed": run_id[3],
                        "user": run_id[4],
                        "tau": run_id[5],
                        "epsilon": run_id[6],
                        "epochs": spec["epochs"],
                    },
                },
                best_path,
            )
        else:
            stale += 1
        atomic_torch_save(
            {
                "run_id": run_id,
                "epoch": epoch,
                "model": model_state(model),
                "optimizer": optimizer.state_dict(),
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "stale": stale,
                "best_weights": best_weights,
                "history": history,
            },
            latest,
        )
        atomic_json(folder / "history.json", history)
        if spec.get("patience") is not None and stale >= spec["patience"]:
            break
    model.load_state_dict(best_weights)
    return best_path


def self_test(cfg):
    x = torch.randn(2, 20, 50, 56)
    if padded_control(x).shape != x.shape or bmpd(x, cfg["tau"], cfg["epsilon"]).shape != x.shape:
        raise AssertionError("Representation shape mismatch")
    constant = torch.ones_like(x)
    if torch.count_nonzero(padded_control(constant)) or torch.count_nonzero(
        bmpd(constant, cfg["tau"], cfg["epsilon"])
    ):
        raise AssertionError("Constant-window check failed")
    probe = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    if reference_scale(probe).item() != 2.0:
        raise AssertionError("Lower-median reference check failed")
    fallback = torch.tensor([[[[0.0, 0.0], [0.0, 4.0]]]])
    if reference_scale(fallback).item() != 1.0:
        raise AssertionError("Mean-fallback check failed")
    for backbone in ("litecnn", "resnet18"):
        set_seed(42)
        first = build_model(backbone, "framediff", cfg["tau"], cfg["epsilon"])
        set_seed(42)
        second = build_model(backbone, "bmpd", cfg["tau"], cfg["epsilon"])
        if sum(v.numel() for v in first.parameters()) != PARAMETERS[backbone]:
            raise AssertionError("Parameter count mismatch")
        if any(
            not torch.equal(value, second.state_dict()[name])
            for name, value in first.state_dict().items()
        ):
            raise AssertionError("Paired initialization mismatch")
        first.train()
        loss = F.cross_entropy(first(x[:1]), torch.tensor([0]))
        loss.backward()
        if not torch.isfinite(loss) or any(
            value.grad is not None and not torch.isfinite(value.grad).all()
            for value in first.parameters()
        ):
            raise AssertionError("Forward or gradient check failed")
    for user in range(1, 9):
        train_indices, val_indices, test_indices = loso_split(user)
        if user in train_indices[:, 0] or user in val_indices[:, 0] or set(test_indices[:, 0]) != {user}:
            raise AssertionError("Held-out-user leakage")
    train_indices, _, _ = loso_split(1)
    sampler = SevenUserClassBatchSampler(train_indices, 42)
    sampler.set_epoch(1)
    positions = [index for batch in sampler for index in batch]
    if len(positions) != len(set(positions)) or len(positions) != len(train_indices):
        raise AssertionError("Sampler coverage check failed")
    print("Self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--mode", choices=("train", "test"))
    parser.add_argument("--backbones", nargs="+", choices=("litecnn", "resnet18"))
    parser.add_argument("--methods", nargs="+", choices=("framediff", "bmpd"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--users", nargs="+", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if args.self_test:
        self_test(cfg)
        return
    if args.mode:
        cfg["mode"] = args.mode
    if cfg["mode"] not in ("train", "test"):
        raise ValueError("mode must be train or test")
    if cfg["num_workers"] != 0 or cfg["tau"] <= 0 or cfg["epsilon"] <= 0:
        raise ValueError("The released protocol uses num_workers=0 and positive tau and epsilon")
    full_backbones, full_methods = cfg["backbones"], cfg["methods"]
    full_seeds, full_users = cfg["seeds"], cfg["users"]
    backbones = args.backbones or full_backbones
    methods = args.methods or full_methods
    seeds = args.seeds or full_seeds
    users = args.users or full_users
    for values in (full_backbones, full_methods, full_seeds, full_users):
        if not values or len(set(values)) != len(values):
            raise ValueError("Configuration lists must be nonempty and unique")
    if set(full_backbones) - set(PARAMETERS) or set(full_methods) - {"bmpd", "framediff"}:
        raise ValueError("Unsupported backbone or method")
    if set(full_users) - set(range(1, 9)):
        raise ValueError("Users must be between 1 and 8")
    if not set(backbones) <= set(full_backbones) or not set(methods) <= set(full_methods):
        raise ValueError("Command-line selection is outside the configured protocol")
    if not set(seeds) <= set(full_seeds) or not set(users) <= set(full_users):
        raise ValueError("Command-line selection is outside the configured protocol")
    for backbone in backbones:
        spec = cfg[backbone]
        if spec["batch_size"] != 32 or spec["epochs"] < 1 or spec["learning_rate"] <= 0:
            raise ValueError("Invalid backbone training settings")
        if backbone == "litecnn" and spec.get("patience") is not None:
            raise ValueError("LiteCNN uses the fixed 30-epoch schedule without early stopping")
        if backbone == "resnet18" and spec.get("patience") != 20:
            raise ValueError("ResNet-18 uses early-stopping patience 20")
    paths = {}
    for key in ("data_dir", "cache_dir", "output_dir", "checkpoint_dir"):
        path = Path(cfg[key]).expanduser()
        paths[key] = path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
    output = paths["output_dir"] / cfg["mode"]
    if output == paths["checkpoint_dir"] or paths["checkpoint_dir"] in output.parents:
        raise ValueError("Keep evaluation outputs separate from supplied checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(cfg["threads"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set device: cpu for CPU evaluation.")
    prepare_data(paths["data_dir"], paths["cache_dir"])
    data_hashes = {
        str(user): json.loads((paths["cache_dir"] / f"user{user}.json").read_text())[
            "cache_sha256"
        ]
        for user in range(1, 9)
    }
    settings = {
        key: cfg[key]
        for key in (
            "mode",
            "tau",
            "epsilon",
            "backbones",
            "methods",
            "seeds",
            "users",
            "num_workers",
            "threads",
            "device",
            "litecnn",
            "resnet18",
        )
    }
    settings.update(data_hashes=data_hashes, torch=torch.__version__, numpy=np.__version__)
    settings["code_hashes"] = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("main.py", "model.py", "data.py", "utils.py")
    }
    manifest = output / "settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Settings differ from existing outputs. Choose a new output_dir.")
    atomic_json(manifest, settings)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    common = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    for backbone in backbones:
        spec = cfg[backbone]
        for seed in seeds:
            for user in users:
                train_indices, val_indices, test_indices = loso_split(user)
                for method in methods:
                    folder = output / backbone / method / f"seed{seed}" / f"user{user}"
                    folder.mkdir(parents=True, exist_ok=True)
                    checkpoint = (
                        paths["checkpoint_dir"]
                        / backbone
                        / method
                        / f"seed{seed}"
                        / f"user{user}"
                        / "best.pth"
                    )
                    if cfg["mode"] == "train":
                        checkpoint = folder / "best.pth"
                    result_path = folder / "result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        if not checkpoint.exists() or result["checkpoint_sha256"] != sha256(checkpoint):
                            raise ValueError("Checkpoint changed since evaluation")
                        print(
                            f"Skipping completed {backbone} {method}, seed {seed}, U{user}",
                            flush=True,
                        )
                        continue
                    print(
                        f"{cfg['mode']}: {backbone} {method}, seed {seed}, U{user}",
                        flush=True,
                    )
                    test_loader = DataLoader(
                        PPBPDataset(paths["cache_dir"], test_indices),
                        batch_size=spec["batch_size"],
                        shuffle=False,
                        **common,
                    )
                    set_seed(seed)
                    model = build_model(
                        backbone, method, cfg["tau"], cfg["epsilon"]
                    ).to(device)
                    if sum(v.numel() for v in model.parameters()) != PARAMETERS[backbone]:
                        raise AssertionError("Unexpected parameter count")
                    if cfg["mode"] == "train":
                        sampler = SevenUserClassBatchSampler(train_indices, seed)
                        train_loader = DataLoader(
                            PPBPDataset(paths["cache_dir"], train_indices),
                            batch_sampler=sampler,
                            **common,
                        )
                        val_loader = DataLoader(
                            PPBPDataset(paths["cache_dir"], val_indices),
                            batch_size=spec["batch_size"],
                            shuffle=False,
                            **common,
                        )
                        run_id = [
                            signature,
                            backbone,
                            method,
                            seed,
                            user,
                            cfg["tau"],
                            cfg["epsilon"],
                        ]
                        checkpoint = train(
                            model,
                            train_loader,
                            val_loader,
                            sampler,
                            folder,
                            spec,
                            seed,
                            run_id,
                            device,
                        )
                        del train_loader, val_loader
                    else:
                        if not checkpoint.exists():
                            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                        saved = load_torch(checkpoint)
                        metadata = saved.get("metadata", {})
                        expected = {
                            "backbone": backbone,
                            "method": method,
                            "seed": seed,
                            "user": user,
                            "tau": cfg["tau"],
                            "epsilon": cfg["epsilon"],
                        }
                        if any(metadata.get(key) != value for key, value in expected.items()):
                            raise ValueError("Checkpoint metadata does not match configuration")
                        model.load_state_dict(saved["model_state"], strict=True)
                    metrics, artifacts = evaluate(model, test_loader, device, keep=True)
                    for name, value in artifacts.items():
                        np.save(folder / f"{name}.npy", value.numpy())
                    np.save(folder / "test_indices.npy", test_indices)
                    result = {
                        "backbone": backbone,
                        "method": method,
                        "seed": seed,
                        "user": user,
                        "accuracy": metrics["top1"],
                        "macro_f1": metrics["macro_f1"],
                        "checkpoint_sha256": sha256(checkpoint),
                    }
                    atomic_json(result_path, result)
                    summarize(
                        output, full_backbones, full_methods, full_seeds, full_users
                    )
                    del model, test_loader, artifacts
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    summarize(output, full_backbones, full_methods, full_seeds, full_users)


if __name__ == "__main__":
    main()
