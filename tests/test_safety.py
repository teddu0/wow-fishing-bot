import threading
import time
from types import SimpleNamespace

import pytest

from wow_fishing.safety import GuardedInput, Stopped, StopToken, Watchdog


class Device:
    def __init__(self):
        self.events = []
        self.position = None

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))


def test_stopped_token_prevents_all_input():
    token, device = StopToken(), Device()
    controller = GuardedInput(token, lambda: None, device, device, "1", "right", (100, 200))
    token.stop("F8")
    for action in (controller.cast, lambda: controller.loot((10, 20))):
        with pytest.raises(Stopped):
            action()
    assert device.events == []
    assert device.position is None


def test_focus_loss_between_move_and_click_prevents_press():
    token, device = StopToken(), Device()
    calls = 0

    def validate():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Другое окно")

    controller = GuardedInput(token, validate, device, device, "1", "right", (-100, 200))
    with pytest.raises(Stopped):
        controller.loot((10, 20))
    assert device.position == (-90, 220)
    assert device.events == []


def test_key_is_released_when_f8_interrupts_press():
    token, device = StopToken(), Device()
    keyboard = SimpleNamespace(press=lambda key: token.stop("F8"), release=device.release)
    controller = GuardedInput(token, lambda: None, keyboard, device, "1", "right", (0, 0))
    controller.cast()
    assert device.events == [("release", "1")]


def test_watchdog_runs_while_vision_is_blocked():
    token = StopToken()
    lost = threading.Event()

    def validate():
        if lost.is_set():
            raise RuntimeError("Фокус потерян")

    watchdog = Watchdog(token, validate)
    watchdog.start()
    try:
        before = time.monotonic()
        lost.set()
        assert token.event.wait(0.25)
        assert time.monotonic() - before < 0.25
    finally:
        watchdog.close()
