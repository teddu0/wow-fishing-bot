from __future__ import annotations

import threading
from collections.abc import Callable


class Stopped(RuntimeError):
    pass


class StopToken:
    def __init__(self):
        self.event = threading.Event()
        self.reason = ""
        self._lock = threading.Lock()

    def stop(self, reason: str) -> None:
        with self._lock:
            if not self.event.is_set():
                self.reason = reason
                self.event.set()

    def check(self) -> None:
        if self.event.is_set():
            raise Stopped(self.reason)


class Watchdog:
    """Focus/identity checks are independent of capture, vision and OCR."""

    def __init__(self, token: StopToken, validate: Callable[[], None]):
        self.token = token
        self.validate = validate
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name="window-guard", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.closed.wait(0.02) and not self.token.event.is_set():
            try:
                self.validate()
            except Exception as exc:
                self.token.stop(str(exc))

    def close(self) -> None:
        self.closed.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1)


class GuardedInput:
    """Checks immediately before each input; releases pressed keys even on cancellation."""

    def __init__(
        self,
        token: StopToken,
        validate: Callable[[], None],
        keyboard,
        mouse,
        key,
        right_button,
        origin: tuple[int, int],
    ):
        self.token, self.validate = token, validate
        self.keyboard, self.mouse = keyboard, mouse
        self.key, self.right_button, self.origin = key, right_button, origin

    def _check(self) -> None:
        self.token.check()
        try:
            self.validate()
        except Exception as exc:
            self.token.stop(str(exc))
            raise Stopped(str(exc)) from exc
        self.token.check()

    def cast(self) -> None:
        self._check()
        try:
            self.keyboard.press(self.key)
            self.token.event.wait(0.035)
        finally:
            self.keyboard.release(self.key)

    def loot(self, point: tuple[int, int]) -> None:
        self._check()
        self.mouse.position = (self.origin[0] + point[0], self.origin[1] + point[1])
        self._check()
        try:
            self.mouse.press(self.right_button)
            self.token.event.wait(0.035)
        finally:
            self.mouse.release(self.right_button)
