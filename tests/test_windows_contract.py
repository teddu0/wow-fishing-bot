import sys
from types import SimpleNamespace

import pytest

from wow_fishing.model import Rect, Window
from wow_fishing.windows import WindowGuard, monitor_region


def test_secondary_monitor_physical_coordinates():
    assert monitor_region(Rect(-1800, 50, 1200, 800), (-1920, 0, 0, 1080)) == (120, 50, 1320, 850)
    with pytest.raises(RuntimeError):
        monitor_region(Rect(-100, 50, 1200, 800), (-1920, 0, 0, 1080))


@pytest.mark.parametrize("change", ["closed", "pid", "started", "focus", "geometry", "minimized"])
def test_guard_rejects_changed_window(monkeypatch, change):
    import wow_fishing.windows as windows

    state = dict(exists=True, pid=1, started=100.0, foreground=123, width=800, minimized=False)
    gui = SimpleNamespace(
        IsWindow=lambda _: state["exists"],
        IsWindowVisible=lambda _: True,
        IsIconic=lambda _: state["minimized"],
        GetClientRect=lambda _: (0, 0, state["width"], 600),
        ClientToScreen=lambda *_: (50, 60),
        GetForegroundWindow=lambda: state["foreground"],
    )
    process = SimpleNamespace(GetWindowThreadProcessId=lambda _: (1, state["pid"]))
    monkeypatch.setitem(sys.modules, "win32gui", gui)
    monkeypatch.setitem(sys.modules, "win32process", process)
    monkeypatch.setattr(windows, "process_info", lambda _: ("wow.exe", state["started"]))
    guard = WindowGuard(Window(123, 1, "WoW", 100.0))
    guard.validate()
    changes = {
        "closed": ("exists", False),
        "pid": ("pid", 2),
        "started": ("started", 200.0),
        "focus": ("foreground", 456),
        "geometry": ("width", 900),
        "minimized": ("minimized", True),
    }
    key, value = changes[change]
    state[key] = value
    with pytest.raises(RuntimeError):
        guard.validate()
