"""Read the FSD50K ground-truth layout verified on ailab01.
读取FSD50K的dev.csv，类别mid，音频路径黄金二官方的train/val划分
dev.csv: fname,labels,mids,split (train/val)
eval.csv: fname,labels,mids (official eval partition)
vocabulary.csv: headerless integer index,label,MID
MID columns are authoritative; names are only checked against the vocabulary.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from datasets.paths.audioset_paths import load_class_map


def _csv_list(value, field, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: empty {field}")
    values = [item.strip() for item in value.split(",")]
    if any(not item for item in values):
        raise ValueError(f"{context}: empty item in {field}")
    return values


def _read_vocabulary(path):
    by_mid, by_label, indices = {}, {}, set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if len(row) != 3 or not row[0].strip().isdigit():
                raise ValueError(f"{path}:{line}: expected headerless index,label,MID vocabulary")
            index, label, mid = (value.strip() for value in row)
            if not label or not mid.startswith(("/m/", "/t/")):
                raise ValueError(f"{path}:{line}: invalid vocabulary label or MID")
            if int(index) in indices or label in by_label or mid in by_mid:
                raise ValueError(f"{path}:{line}: duplicate vocabulary index, label or MID")
            indices.add(int(index))
            by_mid[mid], by_label[label] = label, mid
    if not by_mid or indices != set(range(len(indices))):
        raise ValueError(f"{path}: vocabulary indices must be contiguous from zero")
    return by_mid


def build_fsd50k_index(dev_audio_dir, eval_audio_dir, ground_truth_dir, class_map_path, *, strict=False):
    """Return official 'train'/'val'/'eval' entries and an availability report.

    All IDs are checked for duplicates across all CSV rows before filtering,
    including missing files and unmapped classes. Never infer or randomize splits.
    Available means the referenced file exists, not that its waveform was decoded.
    """
    mid_to_index, names = load_class_map(class_map_path)
    if len(names) != 527 or len(mid_to_index) != 527 or set(mid_to_index.values()) != set(range(527)):
        raise ValueError("Expected the existing AudioSet 527-class map, with indices 0..526")
    ground_truth = Path(ground_truth_dir).expanduser().resolve()
    audio_dirs = {"dev": Path(dev_audio_dir).expanduser().resolve(),
                  "eval": Path(eval_audio_dir).expanduser().resolve()}
    for directory in (ground_truth, *audio_dirs.values()):
        if not directory.is_dir():
            raise FileNotFoundError(f"FSD50K directory does not exist: {directory}")
    vocabulary = _read_vocabulary(ground_truth / "vocabulary.csv")
    splits = {split: [] for split in ("train", "val", "eval")}
    counts = {split: Counter() for split in splits}
    unmapped = {split: Counter() for split in splits}
    missing = {split: [] for split in splits}
    seen_ids, seen_paths, schema = {}, {}, {}
    for source in ("dev", "eval"):
        path = ground_truth / f"{source}.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"fname", "labels", "mids"} | ({"split"} if source == "dev" else set())
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: expected fields {sorted(required)}; got {reader.fieldnames}. "
                                 "Official MID and split metadata are required; no inferred split is allowed.")
            schema[source] = list(reader.fieldnames)
            for line, row in enumerate(reader, 2):
                context = f"{path}:{line}"
                fsd_id = (row.get("fname") or "").strip()
                if not fsd_id.isascii() or not fsd_id.isdigit() or fsd_id != str(int(fsd_id)):
                    raise ValueError(f"{context}: fname must be a canonical numeric FSD recording ID")
                split = (row.get("split") or "").strip() if source == "dev" else "eval"
                if source == "dev" and split not in {"train", "val"}:
                    raise ValueError(f"{context}: unknown official dev split {split!r}; expected train/val")
                if source == "eval" and "split" in row and row["split"] != "eval":
                    raise ValueError(f"{context}: conflicting eval partition metadata")
                if fsd_id in seen_ids:
                    raise ValueError(f"{context}: duplicate FSD ID {fsd_id} in {seen_ids[fsd_id]} and {split}")
                seen_ids[fsd_id] = split
                labels = _csv_list(row.get("labels"), "labels", context)
                original_mids = _csv_list(row.get("mids"), "mids", context)
                if len(labels) != len(original_mids):
                    raise ValueError(f"{context}: labels and mids have different lengths")
                for label, mid in zip(labels, original_mids):
                    if vocabulary.get(mid) != label:
                        raise ValueError(f"{context}: label/MID pair {label!r}, {mid!r} disagrees with vocabulary")
                    if mid not in mid_to_index:
                        unmapped[split][(label, mid)] += 1
                mids = sorted(set(original_mids) & mid_to_index.keys())
                counts[split]["metadata_rows"] += 1
                audio = (audio_dirs[source] / f"{fsd_id}.wav").resolve()
                if audio in seen_paths:
                    raise ValueError(f"{context}: duplicate audio path {audio}; IDs {seen_paths[audio]} and {fsd_id}")
                seen_paths[audio] = fsd_id
                if not mids:
                    counts[split]["no_mapped_mid_count"] += 1
                if not audio.is_file():
                    missing[split].append(fsd_id)
                    continue
                counts[split]["existing_audio_count"] += 1
                if not mids:
                    continue
                splits[split].append({
                    "audio_path": str(audio), "fsd_id": fsd_id, "split": split,
                    "mids": mids, "class_indices": [mid_to_index[mid] for mid in mids],
                    "original_labels": labels, "original_mids": original_mids,
                })
    if strict and any(missing.values()):
        raise FileNotFoundError(f"Missing FSD50K audio: { {k: len(v) for k, v in missing.items()} }")
    if any(not counts[split]["metadata_rows"] for split in splits):
        raise ValueError("Official FSD50K train, val and eval partitions must all contain metadata rows")
    reports = {}
    all_unmapped = Counter()
    for split, entries in splits.items():
        entries.sort(key=lambda e: int(e["fsd_id"]))
        all_unmapped.update(unmapped[split])
        covered = sorted({mid for entry in entries for mid in entry["mids"]})
        reports[split] = {
            "metadata_rows": counts[split]["metadata_rows"],
            "existing_audio_count": counts[split]["existing_audio_count"],
            "usable_audio_count": len(entries), "mapped_mid_count": len(covered), "mapped_mids": covered,
            "no_mapped_mid_count": counts[split]["no_mapped_mid_count"],
            "missing_audio_count": len(missing[split]), "missing_audio_examples": missing[split][:10],
        }
    covered = sorted({mid for entries in splits.values() for entry in entries for mid in entry["mids"]})
    return {"splits": splits, "report": {
        "ground_truth_dir": str(ground_truth), "schema": schema,
        "vocabulary_format": "headerless index,label,MID", "vocabulary_mid_count": len(vocabulary),
        "vocabulary_mapped_mid_count": len(vocabulary.keys() & mid_to_index.keys()),
        "splits": reports, "usable_audio_count": sum(len(v) for v in splits.values()),
        "mapped_mid_count": len(covered), "mapped_mids": covered,
        "unmapped_labels": [{"label": label, "mid": mid, "annotation_count": count}
                            for (label, mid), count in sorted(all_unmapped.items())],
        "split_overlap_count": 0,
    }}


def read_fsd_dev_index(dev_audio_dir, ground_truth_dir, class_map_path):
    """Only official train/val rows in dev.csv; never opens another partition."""
    import csv
    mid_to_index,names=load_class_map(str(class_map_path))
    if len(names)!=527 or len(mid_to_index)!=527 or set(mid_to_index.values())!=set(range(527)):
        raise ValueError('Expected existing AudioSet 527-class mapping')
    directory=Path(dev_audio_dir).resolve();ground_truth=Path(ground_truth_dir).resolve()
    if not directory.is_dir() or not ground_truth.is_dir():raise FileNotFoundError('Missing FSD dev/ground-truth directory')
    vocabulary=_read_vocabulary(ground_truth/'vocabulary.csv')
    entries={part:[] for part in ('train','val')}
    counts={part:Counter() for part in entries}
    seen_ids={};seen_paths={}
    path=ground_truth/'dev.csv'
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'fname','labels','mids','split'}.issubset(reader.fieldnames or []):
            raise ValueError('dev.csv must contain fname,labels,mids,split')
        for line,row in enumerate(reader,2):
            context=f'{path}:{line}'
            fsd_id=(row.get('fname') or '').strip();part=(row.get('split') or '').strip()
            if not fsd_id.isascii() or not fsd_id.isdigit() or fsd_id!=str(int(fsd_id)):
                raise ValueError(f'{context}: invalid FSD ID')
            if part not in entries:raise ValueError(f'{context}: invalid official dev partition {part!r}')
            if fsd_id in seen_ids:raise ValueError(f'Duplicate FSD ID {fsd_id}: {seen_ids[fsd_id]} / {part}')
            seen_ids[fsd_id]=part
            labels=_csv_list(row.get('labels'),'labels',context)
            original_mids=_csv_list(row.get('mids'),'mids',context)
            if len(labels)!=len(original_mids):raise ValueError(f'{context}: labels/MIDs length mismatch')
            if any(vocabulary.get(mid)!=label for label,mid in zip(labels,original_mids)):
                raise ValueError(f'{context}: label/MID vocabulary mismatch')
            mids=sorted(set(original_mids)&mid_to_index.keys())
            audio=(directory/f'{fsd_id}.wav').resolve()
            if audio in seen_paths:raise ValueError(f'Duplicate FSD audio path: {audio}')
            seen_paths[audio]=fsd_id
            counts[part]['metadata_rows']+=1
            if not mids:counts[part]['no_mapped_mid_count']+=1
            if not audio.is_file():raise FileNotFoundError(f'Missing FSD dev audio: {audio}')
            if mids:
                entries[part].append({'fsd_id':fsd_id,'split':part,'audio_path':str(audio),'mids':mids,
                                     'class_indices':[mid_to_index[m] for m in mids]})
    if any(not counts[p]['metadata_rows'] for p in entries):raise ValueError('Need both official train and val rows')
    for part in entries:entries[part].sort(key=lambda e:int(e['fsd_id']))
    return entries,{part:dict(counts[part]) for part in entries}
