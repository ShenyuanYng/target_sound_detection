"""Persist an 85/15 video-grouped split of official Strong Train."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def video_id(entry):
    return entry.get("ytid") or entry["id"].rsplit("_", 1)[0]


def validate_split_config(hparams):
    """Never silently apply a new holdout to an already-trained old model."""
    required = ("train_val_fraction", "train_split_seed", "train_split_path",
                "test_audio_dir", "test_weak_manifest", "test_strong_manifest")
    if any(not hasattr(hparams, name) for name in required):
        raise ValueError(
            "Legacy eval-split configuration/checkpoint is incompatible with the "
            "Strong Train 85/15 split. Start a new experiment from scratch with "
            "the current configuration; do not resume the old checkpoint."
        )


def split_train(entries, data_root, split_path, fraction=0.15, seed=7):
    """Hold out round(fraction * videos) videos; never split their clips.

    Hash ordering is deterministic regardless of manifest order or Python RNG
    state. Fractions refer to video counts, not clips or expanded conditions.
    """
    if not 0 < fraction < 1:
        raise ValueError("train_val_fraction must be between 0 and 1")
    path = Path(split_path)
    if not path.is_absolute():
        path = Path(data_root) / path
    ids = sorted(e["id"] for e in entries)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs in Strong Train")
    groups = {video_id(e) for e in entries}
    if len(groups) < 2:
        raise ValueError("Strong Train requires at least two independent videos")
    settings = {"strategy": "strong_train_video_holdout_v1",
                "seed": seed, "val_fraction": fraction}
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved.get(key) != value for key, value in settings.items()):
            raise ValueError("Split settings changed; use a new train_split_path")
        if sorted(saved["tr"] + saved["cv"]) != ids:
            raise ValueError("Strong Train inventory changed; use a new train_split_path")
    else:
        ordered = sorted(groups, key=lambda key: (
            hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
        n_val = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
        val_videos = set(ordered[:n_val])
        saved = dict(settings, tr=[], cv=[])
        for entry in entries:
            saved["cv" if video_id(entry) in val_videos else "tr"].append(entry["id"])
        for part in ("tr", "cv"):
            saved[part].sort()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            json.dump(saved, handle, indent=2)
            temporary = handle.name
        os.replace(temporary, path)
    train_ids, val_ids = set(saved["tr"]), set(saved["cv"])
    if train_ids & val_ids or not train_ids or not val_ids:
        raise ValueError("Invalid overlapping or empty train/validation split")
    result = {"tr": [e for e in entries if e["id"] in train_ids],
              "cv": [e for e in entries if e["id"] in val_ids]}
    if {video_id(e) for e in result["tr"]} & {video_id(e) for e in result["cv"]}:
        raise ValueError("A video occurs in both training and validation")
    return result
