from __future__ import annotations

import re
import shutil
import subprocess
import threading
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
import pytesseract

from .model import Image, OcrResult
from .settings import GAME_LANGUAGES, validate_game_language

FULL_PHRASES = {
    "rus": (
        "инвентарь заполнен",
        "сумки заполнены",
        "сумка заполнена",
        "нет места в сумках",
        "недостаточно места в сумках",
        "недостаточно места в инвентаре",
    ),
    "eng": (
        "inventory is full",
        "inventory full",
        "bags are full",
        "bags full",
        "bag is full",
        "not enough room in your bags",
        "not enough space in your bags",
        "not enough room in your inventory",
        "not enough space in your inventory",
    ),
}


def bags_full(text: str, language: str = "rus") -> bool:
    validate_game_language(language)
    normalized = re.sub(r"[^а-яa-z0-9]+", " ", text.lower().replace("ё", "е")).strip()
    return any(f" {phrase} " in f" {normalized} " for phrase in FULL_PHRASES[language])


def check_tesseract(command: str = "", language: str = "rus") -> None:
    validate_game_language(language)
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    elif not shutil.which("tesseract"):
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)
    try:
        result = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        languages = result.stdout.splitlines()
    except Exception as exc:
        raise RuntimeError("Установите Tesseract и укажите путь к tesseract.exe") from exc
    if language not in languages:
        raise RuntimeError(
            f"В Tesseract отсутствует модель {language}.traineddata "
            f"для языка клиента «{GAME_LANGUAGES[language]}»"
        )


def error_text_image(frame: Image) -> Image | None:
    """Find red error text anywhere in the client, preserving spatial line breaks.

    No OCR on arbitrary white chat text: red is the default UIErrorsFrame color.
    Oversized colored scenery is discarded by component shape, not screen position.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 172)
    red &= (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 110)
    mask = red.astype(np.uint8) * 255
    h, w = mask.shape
    scale = max(0.6, h / 1080)
    joined = cv2.dilate(mask, np.ones((max(2, int(3 * scale)), max(8, int(22 * scale))), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (40 * scale <= bw and 7 * scale <= bh <= 65 * scale and bw / bh > 2):
            continue
        crop = mask[max(0, y - 4) : min(h, y + bh + 4), max(0, x - 4) : min(w, x + bw + 4)]
        density = float((crop > 0).mean())
        if 0.015 <= density <= 0.55:
            lines.append((y, x, crop))
    if not lines:
        return None
    if len(lines) > 20:
        raise RuntimeError("Слишком много красного текста для надёжной проверки ошибок")
    lines.sort(key=lambda item: (item[0], item[1]))
    width = max(crop.shape[1] for _, _, crop in lines) + 20
    height = sum(crop.shape[0] + 15 for _, _, crop in lines) + 15
    canvas = np.full((height, width), 255, np.uint8)
    offset = 15
    for _, _, crop in lines:
        canvas[offset : offset + crop.shape[0], 10 : 10 + crop.shape[1]] = 255 - crop
        offset += crop.shape[0] + 15
    return cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


class ErrorReader:
    def __init__(self, language: str = "rus"):
        validate_game_language(language)
        self.language = language

    def read(self, frame: Image) -> tuple[str, bool]:
        prepared = error_text_image(frame)
        if prepared is None:
            return "", False
        text = pytesseract.image_to_string(
            prepared, lang=self.language, config="--psm 6", timeout=3
        )
        return text.strip(), bags_full(text, self.language)


class OcrWorker:
    """One in-flight frame, one latest pending frame; results are never overwritten."""

    def __init__(self, reader: ErrorReader | None = None):
        self.reader = reader or ErrorReader()
        self.pending: Queue[tuple[int, float, Image]] = Queue(maxsize=1)
        self.results: Queue[OcrResult] = Queue()
        self.closed = threading.Event()
        self.sequence = 0
        self.thread = threading.Thread(target=self._run, name="error-ocr", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, frame: Image, captured_at: float) -> None:
        item = (self.sequence, captured_at, frame.copy())
        self.sequence += 1
        try:
            self.pending.put_nowait(item)
        except Full:
            try:
                self.pending.get_nowait()
            except Empty:
                pass
            self.pending.put_nowait(item)

    def drain(self) -> list[OcrResult]:
        results = []
        while True:
            try:
                results.append(self.results.get_nowait())
            except Empty:
                return results

    def _run(self) -> None:
        while not self.closed.is_set():
            try:
                sequence, captured_at, frame = self.pending.get(timeout=0.1)
            except Empty:
                continue
            try:
                text, full = self.reader.read(frame)
                result = OcrResult(sequence, captured_at, text, full)
            except Exception as exc:
                result = OcrResult(sequence, captured_at, "", False, str(exc))
            self.results.put(result)

    def close(self) -> None:
        self.closed.set()
        if self.thread.is_alive():
            self.thread.join(timeout=4)
