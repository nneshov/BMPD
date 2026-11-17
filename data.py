from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler
from utils import load_torch, atomic_json, sha256

USERS = tuple(range(1, 9))
N_CLASSES = 8
FULL_PER_USER_CLASS = 539
TRAIN_PER_USER_CLASS = 462
VAL_PER_USER_CLASS = 77


def loso_split(heldout):
    if heldout not in USERS:
        raise ValueError("Held-out user must be U1--U8")
    train, validation, test = ([], [], [])
    for user in USERS:
        for cls in range(N_CLASSES):
            base = cls * FULL_PER_USER_CLASS
            if user == heldout:
                test.extend(((user, base + j) for j in range(FULL_PER_USER_CLASS)))
            else:
                train.extend(((user, base + j) for j in range(TRAIN_PER_USER_CLASS)))
                validation.extend(
                    (
                        (user, base + j)
                        for j in range(
                            TRAIN_PER_USER_CLASS, TRAIN_PER_USER_CLASS + VAL_PER_USER_CLASS
                        )
                    )
                )
    arrays = tuple((np.asarray(x, dtype=np.int32) for x in (train, validation, test)))
    train, validation, test = arrays
    if (len(train), len(validation), len(test)) != (25872, 4312, 4312):
        raise AssertionError(
            "Wrong LOSO sizes: %d/%d/%d" % (len(train), len(validation), len(test))
        )
    groups = [set(map(tuple, x.tolist())) for x in arrays]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("Train/validation/test overlap")
    source = set(USERS) - {heldout}
    if (
        set(train[:, 0]) != source
        or set(validation[:, 0]) != source
        or set(test[:, 0]) != {heldout}
    ):
        raise AssertionError("User leakage or incorrect LOSO membership")
    for user in source:
        for cls in range(8):
            offset = cls * 539
            tr = np.sort(train[(train[:, 0] == user) & (train[:, 1] // 539 == cls), 1] - offset)
            va = np.sort(
                validation[(validation[:, 0] == user) & (validation[:, 1] // 539 == cls), 1]
                - offset
            )
            if not np.array_equal(tr, np.arange(462)) or not np.array_equal(
                va, np.arange(462, 539)
            ):
                raise AssertionError("Incorrect contiguous77 partition")
    return arrays


class SevenUserClassBatchSampler(BatchSampler):

    def __init__(self, indices, seed):
        self.indices = np.asarray(indices, dtype=np.int32)
        self.seed = int(seed)
        self.epoch = 0
        self.users = sorted(set((int(x) for x in self.indices[:, 0])))
        if len(self.users) != 7 or len(self.indices) != 25872:
            raise ValueError("Sampler needs seven LOSO source users")
        self.cells = {}
        for user in self.users:
            for cls in range(8):
                positions = np.nonzero(
                    (self.indices[:, 0] == user) & (self.indices[:, 1] // 539 == cls)
                )[0]
                if len(positions) != 462:
                    raise AssertionError("Unbalanced sampler cell")
                self.cells[user, cls] = positions

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return 809

    def __iter__(self):
        rng = np.random.RandomState(self.seed + 1009 * self.epoch)
        streams = {}
        for cls in range(8):
            order = list(self.users)
            rng.shuffle(order)
            pools = {u: rng.permutation(self.cells[u, cls]) for u in order}
            stream = []
            for j in range(462):
                for user in order:
                    stream.append(int(pools[user][j]))
            if len(stream) != 3234:
                raise AssertionError("Bad class stream")
            streams[cls] = stream
        seen = []
        for batch_index in range(809):
            lo = 4 * batch_index
            hi = min(lo + 4, 3234)
            batch = []
            for cls in range(8):
                batch.extend(streams[cls][lo:hi])
            expected = 32 if batch_index < 808 else 16
            if len(batch) != expected:
                raise AssertionError("Wrong batch size")
            actual = self.indices[batch]
            classes = actual[:, 1] // 539
            per_class = 4 if expected == 32 else 2
            for cls in range(8):
                selected = actual[classes == cls]
                if (
                    len(selected) != per_class
                    or len(set((int(u) for u in selected[:, 0]))) != per_class
                ):
                    raise AssertionError("Sampler violated class/user uniqueness")
            rng.shuffle(batch)
            seen.extend(batch)
            yield batch
        if len(seen) != 25872 or len(set(seen)) != 25872:
            raise AssertionError("Sampler coverage is not exact")


def prepare_data(data_dir, cache_dir):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for user in USERS:
        source = data_dir / f"PPBP_data_user{user}.pth"
        labels = data_dir / f"labels_user{user}.pth"
        manifest = {
            "data_sha256": sha256(source),
            "labels_sha256": sha256(labels),
            "selection": "first_539_per_class",
        }
        dp = cache_dir / f"user{user}_data.npy"
        lp = cache_dir / f"user{user}_labels.npy"
        mp = cache_dir / f"user{user}.json"
        if mp.exists() and dp.exists() and lp.exists():
            saved = json.loads(mp.read_text())
            if all(saved.get(k) == v for k, v in manifest.items()):
                if (
                    sha256(dp) == saved["cache_sha256"]
                    and sha256(lp) == saved["cache_labels_sha256"]
                ):
                    continue
        print(f"Preparing U{user}", flush=True)
        x, y = load_torch(source), load_torch(labels)
        if not isinstance(x, torch.Tensor) or tuple(x.shape[1:]) != (20, 50, 56):
            raise ValueError(f"Unexpected PPBP dimensions for U{user}")
        if not isinstance(y, torch.Tensor) or y.ndim != 1 or len(y) != len(x):
            raise ValueError(f"Unexpected labels for U{user}")
        if not torch.equal(y, y.long()) or set(y.tolist()) != set(range(8)):
            raise ValueError("Expected integer class labels 0 through 7")
        y = y.long()
        counts = [int((y == c).sum()) for c in range(8)]
        if min(counts) < 539:
            raise ValueError(f"At least 539 windows per class required, found {counts}")
        selected = torch.cat(
            [torch.nonzero(y == c, as_tuple=False).flatten()[:539] for c in range(8)]
        )
        tmp = dp.with_suffix(".tmp")
        array = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(4312, 20, 50, 56)
        )
        for start in range(0, len(selected), 32):
            part = x[selected[start : start + 32]].float()
            if not torch.isfinite(part).all():
                raise ValueError("Non-finite PPBP data")
            array[start : start + len(part)] = part.numpy()
        array.flush()
        del array, x
        tmp.replace(dp)
        np.save(lp, y[selected].numpy())
        np.save(cache_dir / f"user{user}_source_indices.npy", selected.numpy())
        manifest.update(
            cache_sha256=sha256(dp), cache_labels_sha256=sha256(lp), original_class_counts=counts
        )
        atomic_json(mp, manifest)


class PPBPDataset(Dataset):
    def __init__(self, cache_dir, indices):
        self.cache_dir = Path(cache_dir)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.handles = {}
        self.offsets = {}
        self.labels = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        user, index = map(int, self.indices[position])
        if user not in self.handles:
            path = self.cache_dir / f"user{user}_data.npy"
            array = np.load(path, mmap_mode="r")
            if array.shape != (4312, 20, 50, 56) or array.dtype != np.float32:
                raise ValueError(f"Invalid cache for U{user}")
            self.offsets[user] = int(array.offset)
            del array
            self.handles[user] = path.open("rb")
            self.labels[user] = np.load(self.cache_dir / f"user{user}_labels.npy")
        handle = self.handles[user]
        size = 20 * 50 * 56 * 4
        handle.seek(self.offsets[user] + index * size)
        raw = handle.read(size)
        x = np.frombuffer(raw, dtype=np.float32).reshape(20, 50, 56).copy()
        return torch.from_numpy(x), int(self.labels[user][index]), user

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "handles", {}).values():
            handle.close()
