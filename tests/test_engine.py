from dataclasses import replace
from types import SimpleNamespace

import pytest
from conftest import with_bobber

from wow_fishing.engine import FishingEngine
from wow_fishing.model import OcrResult, Rect, State, Tuning
from wow_fishing.ocr import bags_full
from wow_fishing.recording import RecordingInput


def ocr(engine, sequence, now, full=False, error=""):
    engine.accept_ocr(OcrResult(sequence, now, "", full, error))


def waiting_engine(water):
    inputs = RecordingInput()
    engine = FishingEngine(inputs, 0)
    engine.calibration.template = water[:20, :20]
    engine.state = State.WAITING
    engine.bite = SimpleNamespace(update=lambda *_: (True, True), box=Rect(10, 20, 20, 20))
    return engine, inputs


def test_one_click_and_wait_for_post_loot_ocr(water):
    engine, inputs = waiting_engine(water)
    engine.tick(water, 0.1)
    assert [a["action"] for a in inputs.actions] == ["loot"]
    for now in (0.2, 1.0, 2.2, 3.0):
        engine.tick(water, now)
    assert engine.state == State.LOOTING
    ocr(engine, 0, 0.5)
    ocr(engine, 1, 1.0)  # Completed OCR, but captured before the loot interval ended.
    engine.tick(water, 3.1)
    assert engine.state == State.LOOTING
    ocr(engine, 2, 3.2)
    engine.tick(water, 3.2)
    assert engine.state == State.BASELINE
    assert len(inputs.actions) == 1


@pytest.mark.parametrize(
    ("language", "phrase"), [("rus", "Инвентарь заполнен."), ("eng", "Inventory is full.")]
)
def test_full_bags_stop_before_next_cast(water, language, phrase):
    engine, inputs = waiting_engine(water)
    engine.tick(water, 0.1)
    engine.accept_ocr(OcrResult(0, 0.5, phrase, bags_full(phrase, language)))
    engine.tick(water, 2.2)
    engine.accept_ocr(OcrResult(1, 2.3, phrase, bags_full(phrase, language)))
    for now in (3, 4, 5):
        engine.tick(water, now)
    assert engine.state == State.STOPPED
    assert "Сумки заполнены" in engine.reason
    assert len(inputs.actions) == 1


def test_repeated_or_out_of_order_ocr_is_not_confirmation(water):
    engine, _ = waiting_engine(water)
    ocr(engine, 1, 1, full=True)
    ocr(engine, 1, 1, full=True)
    ocr(engine, 0, 0.5, full=True)
    assert engine.state != State.STOPPED
    ocr(engine, 2, 2, full=False)
    assert engine.full_streak == 0


def test_ocr_failure_or_staleness_stops_all_actions(water):
    for error in ("Tesseract timeout", ""):
        engine, inputs = waiting_engine(water)
        if error:
            ocr(engine, 0, 0, error=error)
        engine.tick(water, 10)
        assert engine.state == State.STOPPED
        assert inputs.actions == []


def test_three_failed_casts_stop(water):
    inputs = RecordingInput()
    engine = FishingEngine(inputs, 0)
    engine.calibration.template = water[:20, :20]
    for step in range(1200):
        now = step / 10
        ocr(engine, step, now)
        engine.tick(water, now)
        if engine.state == State.STOPPED:
            break
    assert engine.state == State.STOPPED
    assert engine.casts == 3
    assert "Три неудачные" in engine.reason


def test_calibration_fails_after_at_most_three_casts(water):
    engine = FishingEngine(RecordingInput(), 0)
    for step in range(1200):
        now = step / 10
        ocr(engine, step, now)
        engine.tick(water, now)
        if engine.state == State.STOPPED:
            break
    assert engine.calibration_casts == 3
    assert engine.state == State.STOPPED


def test_end_to_end_calibration_uses_two_casts_without_clicking(water):
    tuning = replace(Tuning(), cast_timeout=5.0, search_timeout=3.0, settle_time=0.4)
    inputs = RecordingInput()
    engine = FishingEngine(inputs, 0, tuning=tuning)
    for step in range(100):
        now = step / 10
        ocr(engine, step, now)
        show = engine.state == State.SEARCH and now - engine.cast_at >= 0.3
        frame = with_bobber(water, 110 if engine.casts == 1 else 180) if show else water
        engine.tick(frame, now)
        if engine.state == State.WAITING:
            break
    assert engine.state == State.WAITING
    assert engine.calibration.template is not None
    assert [a["action"] for a in inputs.actions] == ["cast", "cast"]
