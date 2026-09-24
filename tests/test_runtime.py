import sys
from types import SimpleNamespace

import pytest

from wow_fishing.model import Window
from wow_fishing.runtime import SessionWorker
from wow_fishing.settings import Settings


@pytest.mark.parametrize(("language", "installed"), [("rus", "eng"), ("eng", "rus")])
def test_live_missing_model_stops_before_capture_and_input(
    tmp_path, monkeypatch, language, installed
):
    monkeypatch.setitem(
        sys.modules, "pythoncom",
        SimpleNamespace(CoInitialize=lambda: None, CoUninitialize=lambda: None),
    )
    listener = SimpleNamespace(stop=lambda: None, join=lambda **_: None)
    monkeypatch.setattr("wow_fishing.windows.start_hotkey", lambda _: listener)
    monkeypatch.setattr(
        "subprocess.run", lambda *_, **__: SimpleNamespace(stdout=f"{installed}\n")
    )

    def unexpected_access(*args, **kwargs):
        pytest.fail("Missing OCR model must block window capture and input")

    for name in ("WindowGuard", "DesktopSource", "make_input"):
        monkeypatch.setattr(f"wow_fishing.windows.{name}", unexpected_access)
    worker = SessionWorker(
        Window(1, 2, "World of Warcraft", 3),
        Settings(game_language=language),
        tmp_path / "session",
    )
    worker.run()
    assert worker.token.event.is_set()
    assert f"{language}.traineddata" in worker.token.reason
