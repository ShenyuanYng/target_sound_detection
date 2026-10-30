from __future__ import annotations

"""
把AudioSet的音频文件，弱标签csv，强标签tsv按照clip_id对齐，最后整理成模型dataset可以使用的数据列表

弱标签：
YTID,start_seconds,end_seconds,positive_labels
abc123,0,10,"/m/09x0r,/m/05zppz"
强标签：
clip_id        onset    offset    MID
abc123_0       1.2      3.5       /m/09x0r
abc123_0       6.1      8.4       /m/05zppz

"""
import csv
import itertools
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}
PathOrPaths = Optional[Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]]

#把配置文件中的相对路径换成完整路径
def resolve_path(path: Optional[str], data_root: str) -> Optional[str]:
    """Resolve configuration paths relative to ``data_root``."""
    if path in (None, "", "none", "null"):
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded
    root = os.path.expandvars(os.path.expanduser(str(data_root)))
    return os.path.join(root, expanded)

#与前者一样，只不过能处理多个路径
def resolve_paths(paths: PathOrPaths, data_root: str) -> List[str]:
    """Resolve one configured path or a YAML list of paths."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        values = [paths]
    else:
        values = list(paths)

    resolved = [resolve_path(str(path), data_root) for path in values]
    return [path for path in resolved if path is not None]



#读取AudioSet的527类类别表,匹配路径和标签
def load_class_map(class_map_path: str) -> Tuple[Dict[str, int], List[str]]:
    """Load AudioSet's ``class_labels_indices.csv``.

    Returns a MID-to-index mapping and display names ordered by class index.
    """
    mid_to_index: Dict[str, int] = {}
    names: Dict[int, str] = {}
    with open(class_map_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"index", "mid", "display_name"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Invalid AudioSet class map {class_map_path!r}; expected columns "
                "index,mid,display_name"
            )
        for row in reader:
            index = int(row["index"])
            mid = row["mid"].strip()
            mid_to_index[mid] = index
            names[index] = row["display_name"].strip()

    if not names:
        raise ValueError(f"AudioSet class map is empty: {class_map_path}")
    max_index = max(names)
    class_names = [names.get(index, "") for index in range(max_index + 1)]
    if any(not name for name in class_names):
        raise ValueError(f"AudioSet class indices must be contiguous: {class_map_path}")
    return mid_to_index, class_names
"""
mid_to_index           
{                      
    "/m/09x0r": 0,
    "/m/05zppz": 1,
    "/m/02zsn": 2
}
class_name:
[
    "Speech",
    "Male speech",
    "Female speech",
    ...
]
"""


#重新生成ytid  
def _clip_id(ytid: str, start_seconds: float) -> str:
    return f"{ytid}_{int(round(start_seconds * 1000.0))}"


def read_weak_labels(path: str, keep_ids: Optional[set] = None) -> List[dict]:
    """Read an official AudioSet segment CSV.

    Extra header columns ``audio_path`` and ``clip_id`` are accepted for custom
    manifests, while the four-column official files work unchanged.
    {
    "id": clip_id,
    "ytid": ytid,
    "clip_start": start,
    "clip_end": end,
    "weak_mids": [...],  里面可能包含 "/m/09x0r", "/m/05zppz"
    "manifest_audio_path": ...
}
    """
    entries: List[dict] = []
    with open(path, newline="", encoding="utf-8") as handle:
        lines = (line for line in handle if not line.lstrip().startswith("#"))
        reader = csv.reader(lines, skipinitialspace=True)
        first_row = next(reader, None)
        if first_row is None:
            return entries

        header = [value.strip() for value in first_row]
        has_header = bool({"YTID", "ytid", "clip_id"}.intersection(header))
        if has_header:
            keys = header
            data_rows = reader
        else:
            keys = ["YTID", "start_seconds", "end_seconds", "positive_labels"]
            data_rows = itertools.chain([first_row], reader)

        for values in data_rows:
            if not values or not values[0].strip():
                continue
            row = {key: (values[i].strip() if i < len(values) else "") for i, key in enumerate(keys)}
            ytid = row.get("YTID", row.get("ytid", "")).strip()
            start = float(row.get("start_seconds", row.get("start", 0.0)) or 0.0)
            end = float(row.get("end_seconds", row.get("end", start + 10.0)) or start + 10.0)
            clip_id = row.get("clip_id", "").strip() or _clip_id(ytid, start)


            if keep_ids is not None and clip_id not in keep_ids:
                continue
            mids = [mid.strip() for mid in row.get("positive_labels", row.get("labels", "")).split(",")]
            entries.append(
                {
                    "id": clip_id,
                    "ytid": ytid,
                    "clip_start": start,
                    "clip_end": end,
                    "weak_mids": [mid for mid in mids if mid],
                    "manifest_audio_path": row.get("audio_path", "").strip(),
                }
            )
    return entries

#读取强标签并转化为字典形式
def read_strong_labels(path: str) -> Dict[str, List[dict]]:
    """Read the official AudioSet strong-label TSV.

    Both the four-column release and the five-column PRESENT/NOT_PRESENT framed
    evaluation file are supported.  Times are relative to each 10-second clip.
    """
    events: Dict[str, List[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            if row[0].strip().lower() in {"clip_id", "segment_id"}:
                continue
            if len(row) < 4:
                raise ValueError(f"Invalid strong-label row in {path!r}: {row!r}")
            clip_id, onset, offset, mid = (value.strip() for value in row[:4])
            status = row[4].strip().upper() if len(row) >= 5 else "PRESENT"
            if status not in {"PRESENT", "NOT_PRESENT"}:
                raise ValueError(f"Unknown strong-label status {status!r} in {path!r}")
            events[clip_id].append(
                {
                    "onset": float(onset),
                    "offset": float(offset),
                    "mid": mid,
                    "present": status == "PRESENT",
                }
            )
    return dict(events)


def _unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))

#根据clip_id/ytid找到真正的wav文件
def _resolve_audio(entry: dict, audio_dirs: Sequence[str]) -> Optional[str]:
    preferred_dirs = [
        path for path in entry.get("preferred_audio_dirs", []) if path in audio_dirs
    ]
    # A paired weak manifest identifies which split a YTID-only WAV belongs to.
    # Falling back to another split can silently select a different 10-second
    # excerpt from the same YouTube video, so keep that association strict.
    search_dirs = preferred_dirs if preferred_dirs else list(audio_dirs)

    manifest_path = entry.get("manifest_audio_path")
    if manifest_path:
        if os.path.isabs(manifest_path) and os.path.isfile(manifest_path):
            return manifest_path
        for audio_dir in search_dirs:
            candidate = os.path.join(audio_dir, manifest_path)
            if os.path.isfile(candidate):
                return candidate

    identifiers = [entry["id"], entry.get("ytid", "")]
    for audio_dir in search_dirs:
        for identifier in identifiers:
            if not identifier:
                continue
            candidates = [os.path.join(audio_dir, identifier)]
            candidates.extend(
                os.path.join(audio_dir, identifier + extension)
                for extension in sorted(AUDIO_EXTENSIONS)
            )
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
    return None
