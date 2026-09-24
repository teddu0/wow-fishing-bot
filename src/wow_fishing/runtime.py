from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from .engine import FishingEngine
from .model import State, Window
from .ocr import ErrorReader, OcrWorker, check_tesseract
from .recording import Recorder
from .replay import replay
from .safety import Stopped, StopToken, Watchdog
from .settings import GAME_LANGUAGES, Settings


def annotated(frame, engine: FishingEngine | None = None):
    image = frame.copy()
    if engine is not None:
        box = (
            engine.bite.box if engine.bite else (engine.detection.box if engine.detection else None)
        )
        if box:
            cv2.rectangle(
                image, (box.x, box.y), (box.x + box.width, box.y + box.height), (161, 203, 82), 2
            )
    # Bound the Qt signal payload and avoid building up full-resolution queued frames.
    if image.shape[1] > 1000:
        image = cv2.resize(image, (1000, round(image.shape[0] * 1000 / image.shape[1])))
    return image


class PreviewWorker(QThread):
    frame = Signal(object)
    failed = Signal(str)

    def __init__(self, window: Window):
        super().__init__()
        self.window = window
        self.token = StopToken()

    def run(self) -> None:
        import pythoncom

        from .windows import DesktopSource, WindowGuard

        source = None
        pythoncom.CoInitialize()
        try:
            guard = WindowGuard(self.window)
            source = DesktopSource(guard)
            while not self.token.event.is_set():
                guard.validate(focus=False)
                frame = source.read()
                if frame is not None:
                    self.frame.emit(annotated(frame))
                self.token.event.wait(0.2)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            try:
                if source:
                    source.close()
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                pythoncom.CoUninitialize()


class SessionWorker(QThread):
    frame = Signal(object)
    status = Signal(str)
    log = Signal(str)
    counts = Signal(int, int)

    def __init__(self, window: Window, settings: Settings, directory: Path):
        super().__init__()
        self.window, self.settings, self.directory = window, settings, directory
        self.token = StopToken()

    def run(self) -> None:
        import pythoncom

        from .windows import DesktopSource, WindowGuard, make_input, start_hotkey

        source = listener = watchdog = ocr = recorder = engine = journal = None
        pythoncom.CoInitialize()
        started = time.monotonic()

        def emit(message: str) -> None:
            self.log.emit(message)
            if journal:
                journal.write(
                    json.dumps(
                        {"time": time.monotonic() - started, "message": message}, ensure_ascii=False
                    )
                    + "\n"
                )
                journal.flush()

        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            journal = (self.directory / "events.jsonl").open("w", encoding="utf-8")
            listener = start_hotkey(self.token)
            self.status.emit("Проверка распознавания текста…")
            check_tesseract(self.settings.tesseract_path, self.settings.game_language)
            emit(f"Язык клиента: {GAME_LANGUAGES[self.settings.game_language]}")
            self.token.check()
            guard = WindowGuard(self.window)
            for seconds in range(5, 0, -1):
                self.status.emit(f"Перейдите в выбранное окно: {seconds} с")
                if self.token.event.wait(1):
                    self.token.check()
            guard.validate()
            watchdog = Watchdog(self.token, guard.validate)
            watchdog.start()
            source = DesktopSource(guard)
            inputs = make_input(guard, self.token, self.settings.cast_key)
            ocr = OcrWorker(ErrorReader(self.settings.game_language))
            ocr.start()
            last_frame = last_ocr = last_preview = time.monotonic()
            last_ocr -= 1
            last_text = ""
            engine = FishingEngine(inputs, last_frame, emit)
            self.status.emit("Подготовка и первая проверка сумок…")
            while not self.token.event.is_set() and engine.state != State.STOPPED:
                loop_at = time.monotonic()
                guard.validate()
                frame = source.read()
                if frame is None:
                    if loop_at - last_frame > 1:
                        raise RuntimeError("DXcam не возвращает кадры")
                    self.token.event.wait(0.02)
                    continue
                captured_at = time.monotonic()
                last_frame = captured_at
                if self.settings.record:
                    if recorder is None:
                        recorder = Recorder(self.directory, frame.shape)
                    recorder.write(frame, captured_at - started)
                if captured_at - last_ocr >= 0.4:
                    ocr.submit(frame, captured_at)
                    last_ocr = captured_at
                for result in ocr.drain():
                    engine.accept_ocr(result)
                    if result.text and result.text != last_text:
                        emit(f"Текст игры: {result.text}")
                    last_text = result.text
                self.token.check()
                if engine.ocr_sequence >= 0:
                    engine.tick(frame, captured_at)
                elif captured_at - engine.ocr_at > engine.tuning.ocr_stale_after:
                    raise RuntimeError("Первая проверка OCR не завершилась")
                if captured_at - last_preview >= 0.2:
                    self.frame.emit(annotated(frame, engine))
                    self.status.emit(engine.state.value)
                    self.counts.emit(engine.casts, engine.loot_attempts)
                    last_preview = captured_at
                if not listener.is_alive():
                    raise RuntimeError("Обработчик F8 завершился; рыбалка остановлена")
                self.token.event.wait(max(0, 0.05 - (time.monotonic() - loop_at)))
            reason = self.token.reason or engine.reason or "Остановлено"
            emit(reason)
            self.status.emit(reason)
        except Stopped as exc:
            emit(str(exc))
            self.status.emit(str(exc))
        except Exception as exc:
            self.token.stop(str(exc))
            emit(f"Ошибка: {exc}")
            self.status.emit(f"Ошибка: {exc}")
            if journal:
                journal.write(
                    json.dumps({"traceback": traceback.format_exc()}, ensure_ascii=False) + "\n"
                )
        finally:
            self.token.stop("Сеанс завершён")
            if watchdog:
                watchdog.close()
            if listener:
                listener.stop()
                listener.join(timeout=1)
            if ocr:
                ocr.close()
            try:
                if engine and engine.calibration.template is not None:
                    cv2.imencode(".png", engine.calibration.template)[1].tofile(
                        str(self.directory / "bobber.png")
                    )
            except Exception as exc:
                self.log.emit(f"Не удалось сохранить образец: {exc}")
            for resource in (recorder, source, journal):
                if resource:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.log.emit(f"Ошибка освобождения ресурса: {exc}")
            pythoncom.CoUninitialize()


class ReplayWorker(QThread):
    frame = Signal(object)
    status = Signal(str)
    log = Signal(str)
    counts = Signal(int, int)

    def __init__(self, video: Path, settings: Settings, report: Path, use_ocr: bool):
        super().__init__()
        self.video, self.settings, self.report, self.use_ocr = video, settings, report, use_ocr
        self.token = StopToken()

    def run(self) -> None:
        last_preview = 0.0

        def on_frame(frame, engine):
            nonlocal last_preview
            if time.monotonic() - last_preview >= 0.1:
                self.frame.emit(annotated(frame, engine))
                self.status.emit(f"Запись · {engine.state.value}")
                self.counts.emit(engine.casts, engine.loot_attempts)
                last_preview = time.monotonic()

        try:
            result = replay(
                self.video,
                token=self.token,
                on_frame=on_frame,
                tesseract_path=self.settings.tesseract_path,
                game_language=self.settings.game_language,
                use_ocr=self.use_ocr,
                emit=self.log.emit,
            )
            self.report.parent.mkdir(parents=True, exist_ok=True)
            self.report.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.log.emit(f"Отчёт сохранён: {self.report}")
            self.status.emit(result["reason"] or "Проверка записи завершена")
        except Exception as exc:
            self.log.emit(f"Ошибка: {exc}")
            self.status.emit(f"Ошибка: {exc}")


def session_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")
