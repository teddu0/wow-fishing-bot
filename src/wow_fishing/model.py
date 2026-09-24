from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]  # Always BGR, physical client pixels.


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def crop(self, image: Image) -> Image:
        return image[self.y : self.y + self.height, self.x : self.x + self.width]


@dataclass(frozen=True)
class Window:
    hwnd: int
    pid: int
    title: str
    process_started: float


@dataclass(frozen=True)
class Detection:
    box: Rect
    confidence: float
    template: Image = field(repr=False, compare=False)


@dataclass(frozen=True)
class OcrResult:
    sequence: int
    captured_at: float
    text: str
    full: bool
    error: str = ""


class State(Enum):
    BASELINE = "Подготовка заброса"
    SEARCH = "Поиск поплавка"
    CALIBRATING = "Проверка поплавка"
    WAITING = "Ожидание поклёвки"
    LOOTING = "Сбор и проверка сумок"
    STOPPED = "Остановлено"


@dataclass(frozen=True)
class Tuning:
    cast_timeout: float = 35.0
    search_timeout: float = 8.0
    settle_time: float = 1.5
    baseline_time: float = 0.6
    loot_wait: float = 2.0
    max_failures: int = 3
    max_calibration_casts: int = 3
    ocr_stale_after: float = 8.0
    match_threshold: float = 0.65
    bite_floor: float = 0.07


DEFAULT_TUNING = Tuning()


class InputController(Protocol):
    def cast(self) -> None: ...

    def loot(self, point: tuple[int, int]) -> None: ...


class FrameSource(Protocol):
    def read(self) -> Image | None: ...

    def close(self) -> None: ...
