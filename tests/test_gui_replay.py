import json
import sys
from types import SimpleNamespace

import cv2
import pytest

from wow_fishing.recording import Recorder
from wow_fishing.replay import replay
from wow_fishing.settings import Settings


@pytest.mark.parametrize("language", ["rus", "eng"])
def test_video_replay_never_needs_windows_or_input(tmp_path, water, monkeypatch, language):
    def unexpected_ocr(*args, **kwargs):
        pytest.fail("Replay without OCR must not call Tesseract")

    monkeypatch.setattr("wow_fishing.replay.check_tesseract", unexpected_ocr)
    monkeypatch.setattr("wow_fishing.ocr.ErrorReader.read", unexpected_ocr)
    recorder = Recorder(tmp_path, water.shape)
    for index in range(30):
        recorder.write(water, index * 0.1)
    recorder.close()
    result = replay(tmp_path / "screen.avi", use_ocr=False, game_language=language)
    assert result["mode"] == "replay-no-input"
    assert result["ocr_enabled"] is False
    assert result["game_language"] == language
    assert result["frames"] == 30
    assert [action["action"] for action in result["actions"]] == ["cast"]


def test_gui_opens_and_closes_without_windows_dependencies(tmp_path):
    from PySide6.QtWidgets import QApplication

    from wow_fishing.app import STYLE, MainWindow

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    window = MainWindow(tmp_path)
    window.show()
    app.processEvents()
    assert not window.start_button.isEnabled()
    assert window.window_combo.currentData() is None
    window.close()
    app.processEvents()


def test_settings_roundtrip_does_not_persist_window_identity(tmp_path):
    path = tmp_path / "settings.json"
    original = Settings("f3", "C:/Tesseract/tesseract.exe", True, game_language="eng")
    original.save(path)
    assert Settings.load(path) == original
    assert "hwnd" not in path.read_text()


def test_old_settings_default_to_russian(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"cast_key": "f3", "record": true}', encoding="utf-8")
    settings = Settings.load(path)
    assert settings.game_language == "rus"
    assert settings.cast_key == "f3"
    assert settings.record is True


@pytest.mark.parametrize("language", ["deu", "", None, [], 1])
def test_invalid_saved_language_is_rejected(tmp_path, language):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"game_language": language}), encoding="utf-8")
    with pytest.raises(ValueError, match="Язык клиента"):
        Settings.load(path)


def test_gui_passes_selected_language_to_replay_and_restores_it(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from wow_fishing.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    workers = []
    monkeypatch.setattr(window, "_launch", workers.append)
    try:
        assert window.language_combo.currentData() == "rus"
        window.language_combo.setCurrentIndex(window.language_combo.findData("eng"))
        window.video = tmp_path / "screen.avi"
        window.start_replay()
        assert workers[0].settings.game_language == "eng"
        assert not window.language_combo.isEnabled()
        window._busy(False)
        assert window.language_combo.isEnabled()
        assert Settings.load(tmp_path / "settings.json").game_language == "eng"
    finally:
        window.close()
        app.processEvents()
    restored = MainWindow(tmp_path)
    try:
        assert restored.language_combo.currentData() == "eng"
        assert restored.window_combo.currentData() is None
    finally:
        restored.close()
        app.processEvents()


def test_replay_worker_uses_english_ocr_and_stops(tmp_path, water, monkeypatch):
    from wow_fishing.runtime import ReplayWorker

    frame = water.copy()
    cv2.putText(frame, "Inventory is full", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    recorder = Recorder(tmp_path, frame.shape)
    try:
        for index in range(30):
            recorder.write(frame, index * 0.1)
    finally:
        recorder.close()
    monkeypatch.setattr("subprocess.run", lambda *_, **__: SimpleNamespace(stdout="eng\n"))

    def recognize(image, *, lang, config, timeout):
        assert lang == "eng"
        return "Inventory is full."

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    report = tmp_path / "report.json"
    worker = ReplayWorker(tmp_path / "screen.avi", Settings(game_language="eng"), report, True)
    worker.run()
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["game_language"] == "eng"
    assert "Сумки заполнены" in result["reason"]
    assert result["actions"] == []


def test_cli_passes_game_language(tmp_path, monkeypatch):
    from wow_fishing.replay import main

    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv",
        ["wow-fishing-replay", "screen.avi", "--game-language", "eng", "--report", str(report)],
    )

    def run_replay(video, **kwargs):
        assert kwargs["game_language"] == "eng"
        assert kwargs["use_ocr"] is True
        return {"game_language": kwargs["game_language"]}

    monkeypatch.setattr("wow_fishing.replay.replay", run_replay)
    main()
    assert json.loads(report.read_text(encoding="utf-8"))["game_language"] == "eng"
