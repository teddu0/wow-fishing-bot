from __future__ import annotations

from collections.abc import Callable

from .model import DEFAULT_TUNING, Image, InputController, OcrResult, State, Tuning
from .vision import BiteDetector, BobberDetector, Calibration


class FishingEngine:
    """Deterministic state machine: timestamps are supplied by live/replay drivers."""

    def __init__(
        self,
        inputs: InputController,
        now: float,
        emit: Callable[[str], None] = lambda _: None,
        tuning: Tuning = DEFAULT_TUNING,
    ):
        self.inputs, self.emit, self.tuning = inputs, emit, tuning
        self.state = State.BASELINE
        self.entered = now
        self.cast_at = now
        self.frames: list[Image] = []
        self.detector = BobberDetector(tuning)
        self.calibration = Calibration(tuning.match_threshold)
        self.bite: BiteDetector | None = None
        self.detection = None
        self.calibration_casts = 0
        self.failures = 0
        self.casts = 0
        self.loot_attempts = 0
        self.ocr_at = now
        self.ocr_sequence = -1
        self.full_streak = 0
        self.loot_ocr_count = 0
        self.reason = ""

    def stop(self, reason: str) -> None:
        if self.state != State.STOPPED:
            self.reason = reason
            self.state = State.STOPPED
            self.emit(reason)

    def _transition(self, state: State, now: float) -> None:
        self.state, self.entered = state, now
        self.emit(state.value)

    def accept_ocr(self, result: OcrResult) -> None:
        if self.state == State.STOPPED or result.sequence <= self.ocr_sequence:
            return
        self.ocr_sequence = result.sequence
        if result.error:
            self.stop(f"Ошибка распознавания текста: {result.error}")
            return
        if result.captured_at < self.ocr_at:
            return
        self.ocr_at = result.captured_at
        self.full_streak = self.full_streak + 1 if result.full else 0
        if self.full_streak >= 2:
            self.stop("Сумки заполнены — рыбалка остановлена")
        if self.state == State.LOOTING and result.captured_at >= self.entered:
            self.loot_ocr_count += 1

    def _baseline(self, now: float) -> None:
        self.frames.clear()
        self.detection = None
        self.bite = None
        self._transition(State.BASELINE, now)

    def _failed(self, now: float, message: str) -> None:
        self.failures += 1
        self.emit(message)
        if self.calibration.template is None:
            if self.calibration_casts >= self.tuning.max_calibration_casts:
                self.stop("Не удалось автоматически определить поплавок за три заброса")
            else:
                self._transition(State.CALIBRATING, now)
        elif self.failures >= self.tuning.max_failures:
            self.stop("Три неудачные попытки подряд — проверьте место и настройки игры")
        else:
            # Wait for the previous cast to expire before collecting a new baseline.
            self._transition(State.CALIBRATING, now)

    def tick(self, frame: Image, now: float) -> None:
        if self.state == State.STOPPED:
            return
        if now - self.ocr_at > self.tuning.ocr_stale_after:
            self.stop("Нет свежей проверки сумок: OCR не успевает или недоступен")
            return
        if self.state == State.BASELINE:
            if len(self.frames) < 9:
                self.frames.append(frame.copy())
            if now - self.entered < self.tuning.baseline_time or len(self.frames) < 3:
                return
            self.detector.begin(self.frames)
            self.frames.clear()
            # A single positive OCR result blocks casting until the next result.
            if self.full_streak:
                return
            self.inputs.cast()
            self.casts += 1
            if self.calibration.template is None:
                self.calibration_casts += 1
            self.cast_at = now
            self._transition(State.SEARCH, now)
        elif self.state == State.SEARCH:
            age = now - self.cast_at
            if age >= self.tuning.search_timeout:
                self._failed(now, "Поплавок не найден однозначно")
                return
            if age < self.tuning.settle_time:
                return
            found = self.detector.find(frame, self.calibration.template)
            if found is None:
                return
            self.detection = found
            if self.calibration.template is None:
                if not self.calibration.observe(found):
                    self.emit(f"Пробный заброс {self.calibration_casts}: кандидат найден")
                    if self.calibration_casts >= self.tuning.max_calibration_casts:
                        self.stop("Кандидаты не совпали между забросами; калибровка не завершена")
                    else:
                        self._transition(State.CALIBRATING, now)
                    return
                self.failures = 0
                self.emit("Автоматическая калибровка завершена")
            self.bite = BiteDetector(frame, found, self.tuning.bite_floor)
            self._transition(State.WAITING, now)
        elif self.state == State.CALIBRATING:
            if now - self.cast_at >= self.tuning.cast_timeout:
                self._baseline(now)
        elif self.state == State.WAITING:
            if now - self.cast_at >= self.tuning.cast_timeout:
                self._failed(now, "Поклёвка не обнаружена до таймаута")
                return
            bitten, visible = self.bite.update(frame, now - self.entered)
            if bitten and not self.full_streak:
                self.inputs.loot(self.bite.box.center)
                self.loot_attempts += 1
                self.loot_ocr_count = 0
                self._transition(State.LOOTING, now)
            elif not visible:
                self._failed(now, "Поплавок потерян; клик отменён")
        elif self.state == State.LOOTING:
            complete = now - self.entered >= self.tuning.loot_wait
            checked = self.ocr_at >= self.entered + self.tuning.loot_wait
            if complete and checked and self.loot_ocr_count >= 2 and not self.full_streak:
                # This is a completed click/check cycle, not proof of an item in the bag.
                self.failures = 0
                self._baseline(now)
