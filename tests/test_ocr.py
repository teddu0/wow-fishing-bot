import threading
from types import SimpleNamespace

import cv2
import pytest

from wow_fishing.ocr import ErrorReader, OcrWorker, bags_full, check_tesseract, error_text_image


@pytest.mark.parametrize(
    "text",
    [
        "Инвентарь заполнен.",
        "Ваши сумки\nзаполнены!",
        "Нет места в сумках",
        "Недостаточно места в инвентаре",
    ],
)
def test_full_messages(text):
    assert bags_full(text)


@pytest.mark.parametrize("text", ["Недостаточно маны", "Цель слишком далеко", "", "сумки"])
def test_other_errors_do_not_stop_as_full(text):
    assert not bags_full(text)


@pytest.mark.parametrize(
    "text",
    [
        "Inventory is full.",
        "Your INVENTORY is\nfull!",
        "Inventory full",
        "Your bags are full.",
        "Bags full!",
        "Your bag is full.",
        "Not enough room in your bags.",
        "Not enough space in your bags.",
        "Not enough room in your inventory.",
        "Not enough space in your inventory.",
    ],
)
def test_english_full_messages(text):
    assert bags_full(text, "eng")
    assert not bags_full(text, "rus")


@pytest.mark.parametrize(
    "text",
    [
        "Not enough mana.",
        "You are too far away.",
        "Your fish got away!",
        "Your bank is full.",
        "You can't carry any more of those items.",
        "Inventory is not full.",
        "Inventory is fuller.",
        "Инвентарь заполнен.",
        "",
    ],
)
def test_other_english_errors_do_not_stop_as_full(text):
    assert not bags_full(text, "eng")


def test_error_regions_found_without_fixed_screen_position(water):
    assert error_text_image(water) is None
    for y in (25, 150, 220):
        frame = water.copy()
        cv2.putText(
            frame, "Inventory full", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 255), 1
        )
        assert error_text_image(frame) is not None


@pytest.mark.parametrize(("language", "installed"), [("rus", "eng"), ("eng", "rus")])
def test_missing_selected_model_blocks_start(monkeypatch, language, installed):
    monkeypatch.setattr(
        "subprocess.run", lambda *_, **__: SimpleNamespace(stdout=f"{installed}\n")
    )
    with pytest.raises(RuntimeError, match=f"{language}.traineddata"):
        check_tesseract(language=language)


@pytest.mark.parametrize("language", ["rus", "eng"])
def test_only_selected_model_is_required(monkeypatch, language):
    monkeypatch.setattr(
        "subprocess.run", lambda *_, **__: SimpleNamespace(stdout=f"{language}\n")
    )
    check_tesseract(language=language)


@pytest.mark.parametrize(
    ("language", "phrase"), [("rus", "Инвентарь заполнен."), ("eng", "Inventory is full.")]
)
def test_reader_uses_selected_model(monkeypatch, water, language, phrase):
    frame = water.copy()
    cv2.putText(frame, "Inventory is full", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    def recognize(image, *, lang, config, timeout):
        assert lang == language
        return phrase

    monkeypatch.setattr("pytesseract.image_to_string", recognize)
    assert ErrorReader(language).read(frame) == (phrase, True)


@pytest.mark.parametrize("language", ["deu", "rus+eng", "", None, []])
def test_unsupported_language_is_rejected(language):
    with pytest.raises(ValueError, match="Язык клиента"):
        ErrorReader(language)
    with pytest.raises(ValueError, match="Язык клиента"):
        check_tesseract(language=language)


def test_worker_does_not_block_submit_and_keeps_result(water):
    entered, release = threading.Event(), threading.Event()

    class Reader:
        def read(self, frame):
            entered.set()
            release.wait(1)
            return "Инвентарь заполнен", True

    worker = OcrWorker(Reader())
    worker.start()
    try:
        worker.submit(water, 1.0)
        assert entered.wait(1)
        worker.submit(water, 2.0)
        worker.submit(water, 3.0)  # Only the latest pending frame is kept.
        release.set()
        first = worker.results.get(timeout=1)
        second = worker.results.get(timeout=1)
        assert first.captured_at == 1.0 and first.full
        assert second.captured_at == 3.0 and second.full
    finally:
        release.set()
        worker.close()
