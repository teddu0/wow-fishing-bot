from __future__ import annotations

import json
from pathlib import Path

import cv2

from .model import Image


class VideoSource:
    def __init__(self, path: Path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Не удалось открыть запись: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 20.0
        self.index = -1

    def read(self) -> Image | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.index += 1
        return frame

    def close(self) -> None:
        self.capture.release()


class Recorder:
    """MJPEG plus real capture timestamps; video FPS alone cannot preserve dropped frames."""

    def __init__(self, directory: Path, shape: tuple[int, ...]):
        h, w = shape[:2]
        self.video = cv2.VideoWriter(
            str(directory / "screen.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (w, h)
        )
        if not self.video.isOpened():
            raise RuntimeError("Не удалось создать запись экрана")
        self.timeline = (directory / "frames.jsonl").open("w", encoding="utf-8")
        self.index = 0

    def write(self, frame: Image, timestamp: float) -> None:
        self.video.write(frame)
        self.timeline.write(json.dumps({"frame": self.index, "time": timestamp}) + "\n")
        self.index += 1

    def close(self) -> None:
        self.video.release()
        self.timeline.close()


class RecordingInput:
    """No OS imports or side effects; also used by state-machine tests."""

    def __init__(self):
        self.actions: list[dict] = []
        self.now = 0.0

    def cast(self) -> None:
        self.actions.append({"time": self.now, "action": "cast"})

    def loot(self, point: tuple[int, int]) -> None:
        self.actions.append({"time": self.now, "action": "loot", "point": point})
