"""Independent weak-only AudioSet index for mix experiments (no Strong TSV)."""
from __future__ import annotations

import os
from collections import Counter

from datasets.paths.audioset_paths import (
    _resolve_audio, load_class_map, read_weak_labels, resolve_path, resolve_paths,
)


def build_audioset_weak_index(audio_dirs, weak_csvs, class_map_path, *, data_root=".", strict=False):
    """Return {'entries': [...], 'report': {...}} using the existing 527 indices.

    Multiple CSVs and directories are paired in order. A single directory may
    hold multiple CSVs. Missing audio is reported (or rejected with strict=True).
    No split is created and no waveform or strong annotation is read.
    """
    mid_to_index, names = load_class_map(resolve_path(str(class_map_path), data_root))
    if len(names) != 527 or len(mid_to_index) != 527 or set(mid_to_index.values()) != set(range(527)):
        raise ValueError("Expected the existing AudioSet 527-class map, with indices 0..526")
    directories = resolve_paths(audio_dirs, data_root)
    manifests = resolve_paths(weak_csvs, data_root)
    if not directories or not manifests:
        raise ValueError("AudioSet audio_dirs and weak_csvs must both be configured")
    if len(directories) != 1 and len(directories) != len(manifests):
        raise ValueError("Pair each AudioSet weak CSV with one audio directory, or use one shared directory")
    for directory in directories:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"AudioSet audio directory does not exist: {directory}")
    for manifest in manifests:
        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"AudioSet weak CSV does not exist: {manifest}")

    by_id, audio_to_id = {}, {}
    unmapped = Counter()
    missing, no_mapped = [], []
    row_count = 0
    for manifest_index, manifest in enumerate(manifests):
        directory = directories[0] if len(directories) == 1 else directories[manifest_index]
        for row in read_weak_labels(manifest):
            row_count += 1
            if not row["ytid"]:
                raise ValueError(f"Missing YouTube video ID in {manifest}: {row['id']}")
            original_mids = sorted(set(row["weak_mids"]))
            unmapped.update(mid for mid in original_mids if mid not in mid_to_index)
            row["preferred_audio_dirs"] = [directory]
            audio = _resolve_audio(row, [directory])
            if audio is None:
                missing.append(row["id"])
                continue
            audio = os.path.realpath(audio)
            mids = sorted(mid for mid in original_mids if mid in mid_to_index)
            if not mids:
                no_mapped.append(row["id"])
                continue
            previous_id = audio_to_id.setdefault(audio, row["id"])
            if previous_id != row["id"]:
                raise ValueError(f"Ambiguous audio {audio}: matches clips {previous_id} and {row['id']}")
            existing = by_id.get(row["id"])
            if existing is not None:
                if existing["audio_path"] != audio or existing["ytid"] != row["ytid"]:
                    raise ValueError(f"Conflicting AudioSet entries for clip {row['id']}")
                mids = sorted(set(existing["weak_mids"]) | set(mids))
                original_mids = sorted(set(existing["original_weak_mids"]) | set(original_mids))
            by_id[row["id"]] = {
                "id": row["id"], "audio_path": audio, "ytid": row["ytid"],
                "weak_mids": mids, "class_indices": [mid_to_index[mid] for mid in mids],
                "original_weak_mids": original_mids,
                "clip_start": row["clip_start"], "clip_end": row["clip_end"],
            }
    if missing and strict:
        raise FileNotFoundError(f"Missing {len(missing)} AudioSet audio files; examples: {missing[:10]}")
    entries = [by_id[key] for key in sorted(by_id)]
    coverage = sorted({mid for entry in entries for mid in entry["weak_mids"]})
    return {"entries": entries, "report": {
        "weak_csvs": manifests, "audio_dirs": directories, "metadata_rows": row_count,
        "background_count": len(entries), "video_count": len({e["ytid"] for e in entries}),
        "mapped_mid_count": len(coverage), "mapped_mids": coverage,
        "missing_audio_count": len(missing), "missing_audio_examples": missing[:10],
        "no_mapped_mid_count": len(no_mapped), "no_mapped_mid_examples": no_mapped[:10],
        "unmapped_mids": dict(sorted(unmapped.items())),
    }}
