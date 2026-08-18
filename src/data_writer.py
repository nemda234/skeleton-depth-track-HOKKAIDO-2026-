"""Ghi dữ liệu skeleton+depth ra .jsonl theo kiểu streaming (append từng dòng),
   để chạy video dài không bị phình RAM."""
import json


class SkeletonDataWriter:
    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8")

    def write_frame(self, frame_idx: int, timestamp_ms: float, people: list):
        record = {
            "frame_idx": frame_idx,
            "timestamp_ms": round(timestamp_ms, 2),
            "people": people,  # list dict như PoseTracker.track() trả về
        }
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
