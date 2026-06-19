import json
import os
import time
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np


JSONL_OFFSET_SUFFIX = ".idx.npy"
JSONL_LOCK_SUFFIX = ".idx.lock"


def is_jsonl_path(path: str) -> bool:
    return str(path).lower().endswith(".jsonl")


def jsonl_offset_path(path: str) -> str:
    return path + JSONL_OFFSET_SUFFIX


def _offset_index_is_fresh(data_path: str, index_path: str) -> bool:
    return (
        os.path.exists(index_path)
        and os.path.getmtime(index_path) >= os.path.getmtime(data_path)
        and os.path.getsize(index_path) > 0
    )


def build_or_load_jsonl_offsets(data_path: str, index_path: Optional[str] = None) -> np.ndarray:
    """Build or load byte offsets for non-empty JSONL records."""
    index_path = index_path or jsonl_offset_path(data_path)
    if _offset_index_is_fresh(data_path, index_path):
        return np.load(index_path, mmap_mode="r")

    lock_path = data_path + JSONL_LOCK_SUFFIX
    lock_fd = None
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
        except FileExistsError:
            if _offset_index_is_fresh(data_path, index_path):
                return np.load(index_path, mmap_mode="r")
            if time.time() - os.path.getmtime(lock_path) > 3600:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
                continue
            time.sleep(2)

    try:
        if _offset_index_is_fresh(data_path, index_path):
            return np.load(index_path, mmap_mode="r")

        offsets = []
        with open(data_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        arr = np.asarray(offsets, dtype=np.int64)
        tmp_path = index_path + f".tmp.{os.getpid()}"
        np.save(tmp_path, arr)
        npy_tmp_path = tmp_path if tmp_path.endswith(".npy") else tmp_path + ".npy"
        os.replace(npy_tmp_path, index_path)
        return np.load(index_path, mmap_mode="r")
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


class JsonlRandomAccessReader:
    """Memory-light random access for JSONL rows using byte offsets."""

    def __init__(self, data_path: str, index_path: Optional[str] = None):
        self.data_path = data_path
        self.index_path = index_path or jsonl_offset_path(data_path)
        self.offsets = build_or_load_jsonl_offsets(data_path, self.index_path)
        self._file = None

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._file is None:
            self._file = open(self.data_path, "rb")
        self._file.seek(int(self.offsets[idx]))
        line = self._file.readline()
        return json.loads(line.decode("utf-8"))

    def __iter__(self) -> Iterator[Dict]:
        for idx in range(len(self)):
            yield self[idx]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        self.close()


def load_json_or_jsonl(path: str):
    if is_jsonl_path(path):
        return JsonlRandomAccessReader(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_or_jsonl(path: str) -> Iterable[Dict]:
    if is_jsonl_path(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    with open(path, "r", encoding="utf-8") as f:
        for row in json.load(f):
            yield row


def row_metadata_for_output(row: Dict, keys: Optional[List[str]] = None) -> Dict:
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("sample_id", row.get("id"))
    for key in keys or []:
        if key in row and key not in metadata:
            metadata[key] = row[key]
    return metadata
