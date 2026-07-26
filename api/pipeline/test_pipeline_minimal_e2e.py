from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_fake_ml_stubs(stub_root: Path) -> None:
    _write_file(
        stub_root / "torch" / "__init__.py",
        """import pickle

float32 = "float32"
long = "long"


class device:
    def __init__(self, kind):
        self.type = kind


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(seed):
        return None


cuda = _Cuda()


def manual_seed(seed):
    return None


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, fn):
        return fn


def no_grad():
    return _NoGrad()


class FakeTensor:
    def __init__(self, shape=(), data=None):
        if isinstance(shape, tuple):
            self.shape = shape
        elif isinstance(shape, list):
            self.shape = tuple(shape)
        elif shape:
            self.shape = (int(shape),)
        else:
            self.shape = ()
        self._data = data

    def to(self, _device):
        return self

    def float(self):
        return self

    def long(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def norm(self, dim=-1, keepdim=True):
        return self

    def __truediv__(self, _other):
        return self

    def __getitem__(self, idx):
        if self.shape and isinstance(idx, int):
            tail = self.shape[1:] if len(self.shape) > 1 else ()
            return FakeTensor(shape=tail)
        return FakeTensor(shape=())

    def numel(self):
        out = 1
        for n in self.shape:
            out *= int(n)
        return out

    def item(self):
        return 0.0

    def any(self):
        return self

    def tolist(self):
        if isinstance(self._data, list):
            return list(self._data)
        if len(self.shape) == 1:
            return list(range(int(self.shape[0])))
        return []


Tensor = FakeTensor


def zeros(shape, device=None, dtype=None):
    return FakeTensor(shape=shape)


def ones(shape, dtype=None):
    if isinstance(shape, tuple):
        target = shape
    else:
        target = (int(shape),)
    return FakeTensor(shape=target)


def ones_like(t):
    return FakeTensor(shape=getattr(t, "shape", ()))


def full_like(t, _fill):
    return FakeTensor(shape=getattr(t, "shape", ()))


def tensor(value, dtype=None):
    if isinstance(value, list):
        return FakeTensor(shape=(len(value),), data=value)
    return FakeTensor(shape=())


def sigmoid(x):
    return x


def softmax(x, dim=-1):
    return x


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path, map_location=None):
    with open(path, "rb") as f:
        return pickle.load(f)


from . import nn  # noqa: E402
from . import optim  # noqa: E402
from . import utils  # noqa: E402
""",
    )
    _write_file(
        stub_root / "torch" / "nn" / "__init__.py",
        """from .. import FakeTensor


class Module:
    def __init__(self):
        pass

    def to(self, _device):
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def parameters(self):
        return []

    def state_dict(self):
        fc = getattr(self, "fc", None)
        if fc is not None and hasattr(fc, "state_dict"):
            return {f"fc.{k}": v for k, v in fc.state_dict().items()}
        return {}

    def load_state_dict(self, _state):
        return self


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = FakeTensor(shape=(self.out_features, self.in_features))
        self.bias = FakeTensor(shape=(self.out_features,))

    def __call__(self, x):
        shape = getattr(x, "shape", ())
        batch = 1
        if isinstance(shape, tuple) and len(shape) > 0:
            try:
                batch = int(shape[0])
            except Exception:
                batch = 1
        return FakeTensor(shape=(batch, self.out_features))

    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}

    def load_state_dict(self, state):
        self.weight = state.get("weight", self.weight)
        self.bias = state.get("bias", self.bias)
        return self


class _Loss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return FakeTensor(shape=())


class BCEWithLogitsLoss(_Loss):
    pass


class CrossEntropyLoss(_Loss):
    pass
""",
    )
    _write_file(
        stub_root / "torch" / "optim" / "__init__.py",
        """class AdamW:
    def __init__(self, params, lr=1e-4, weight_decay=1e-5):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self, set_to_none=True):
        return None

    def step(self):
        return None
""",
    )
    _write_file(
        stub_root / "torch" / "utils" / "__init__.py",
        """from . import data  # noqa: F401
""",
    )
    _write_file(
        stub_root / "torch" / "utils" / "data.py",
        """class Dataset:
    pass


class _Batch:
    def __init__(self, items):
        self.items = list(items)

    def to(self, _device):
        return self


class _IndexList:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ):
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)

    def __len__(self):
        total = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        total = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        for start in range(0, total, self.batch_size):
            end = min(total, start + self.batch_size)
            batch = [self.dataset[i] for i in range(start, end)]
            if not batch:
                continue
            first = batch[0]
            if isinstance(first, tuple) and len(first) == 2:
                xs = [b[0] for b in batch]
                idxs = [b[1] for b in batch]
                yield _Batch(xs), _IndexList(idxs)
            else:
                yield first
""",
    )
    _write_file(
        stub_root / "clip.py",
        """import torch


class _FakeClipModel:
    def eval(self):
        return self

    def float(self):
        return self

    def parameters(self):
        return []

    def encode_image(self, xb):
        batch = 1
        if hasattr(xb, "items"):
            try:
                batch = len(xb.items)
            except Exception:
                batch = 1
        else:
            shape = getattr(xb, "shape", ())
            if isinstance(shape, tuple) and len(shape) > 0:
                try:
                    batch = int(shape[0])
                except Exception:
                    batch = 1
        return torch.zeros((batch, 64))


def load(_name, device=None, jit=False):
    return _FakeClipModel(), (lambda _image: torch.tensor([0.0]))
""",
    )


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode == 0:
        return
    raise AssertionError(
        f"command failed: {' '.join(cmd)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_pipeline_apply_train_infer_with_multilabel_and_single_class_compat(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source_images"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "a.jpg").write_bytes(b"a")
    (source_dir / "b.jpg").write_bytes(b"b")

    exp_dir = tmp_path / "workspaces" / "ws01" / "experiments" / "exp01"
    label_input_dir = exp_dir / "label_input"
    dataset_dir = exp_dir / "dataset"
    label_input_dir.mkdir(parents=True, exist_ok=True)

    labels_path = label_input_dir / "labels.json"
    schema_path = label_input_dir / "label_schema.json"

    labels_path.write_text(
        json.dumps(
            {
                "meta": {"version": "mvp"},
                "items": [
                    {
                        "id": "1",
                        "path": "a.jpg",
                        "labels": {
                            "split": "Train",
                            "character": "alice",
                            "tags": "cool,cute",
                        },
                    },
                    {
                        "id": "2",
                        "path": "b.jpg",
                        "labels": {
                            "split": "val",
                            "character": "bob",
                            "tags": ["cool"],
                        },
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(
            {
                "version": 2,
                "schema_id": "test_v1",
                "heads": [
                    {
                        "id": "character",
                        "type": "single_class",
                        "classes": ["alice", "bob"],
                    },
                    {
                        "id": "tags",
                        "type": "multi_label",
                        "classes": ["cute", "cool"],
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stub_root = tmp_path / "stubs"
    _write_fake_ml_stubs(stub_root)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{stub_root}{os.pathsep}{existing}" if existing else str(stub_root)
    )

    _run(
        [
            sys.executable,
            str(PIPELINE_DIR / "apply_label_mapping.py"),
            "--labels",
            str(labels_path),
            "--root",
            str(source_dir),
            "--dataset-root",
            str(dataset_dir),
            "--schema",
            str(schema_path),
        ],
        env=env,
    )

    index_path = dataset_dir / "index.json"
    assert index_path.exists()
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index_data.get("items", [])) == 2
    first_labels = index_data["items"][0]["labels"]
    assert first_labels["split"] == "train"
    assert first_labels["character"] == "alice"
    assert first_labels["tags"] == ["cute", "cool"]

    _run(
        [
            sys.executable,
            str(PIPELINE_DIR / "train_epoch.py"),
            "--index",
            str(index_path),
            "--schema",
            str(schema_path),
            "--heads",
            "character,tags",
            "--epochs",
            "0",
            "--no-class-weight",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--clip-model",
            "ViT-B/32",
        ],
        env=env,
    )

    models_dir = exp_dir / "models"
    run_dirs = sorted([p for p in models_dir.glob("run_*") if p.is_dir()])
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["clip_model_name"] == "ViT-B/32"
    assert run_meta["heads"] == ["character", "tags"]

    # A1: 学習結果の metrics セクションが run_meta.json に永続化されていること。
    # 本 E2E は epochs=0 の配管スモークなので、値ではなく構造の健全性を検証する。
    assert run_meta.get("run_meta_version") == "1"
    assert "finished_at" in run_meta
    metrics_heads = run_meta.get("metrics", {}).get("heads", {})
    for head_name in ("character", "tags"):
        hm = metrics_heads.get(head_name)
        assert hm is not None, f"metrics missing for head {head_name}"
        assert hm["epochs_trained"] == len(hm["history"])
        assert {"epoch", "val_acc", "val_loss"} <= set(hm["best"].keys())
        for rec in hm["history"]:
            assert {"epoch", "train_loss", "val_loss", "val_acc"} <= set(rec.keys())

    _run(
        [
            sys.executable,
            str(PIPELINE_DIR / "infer_heads.py"),
            "--run-dir",
            str(run_dir),
            "--index",
            str(index_path),
            "--schema",
            str(schema_path),
            "--splits",
            "none",
        ],
        env=env,
    )

    scores_path = exp_dir / "eval" / f"scores_{run_dir.name}.json"
    assert scores_path.exists()
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    head_types = scores.get("meta", {}).get("head_types", {})
    assert head_types.get("character") == "multi_class"
    assert head_types.get("tags") == "multi_label"
    # splits=none => no inference targets. Still, pipeline should complete.
    assert scores.get("items") == []
