import cv2
import numpy as np
from conftest import with_bobber

from wow_fishing.vision import BiteDetector, BobberDetector, Calibration


def find(detector, frame):
    result = None
    for _ in range(4):
        result = detector.find(frame)
    return result


def test_persistent_new_bobber_and_static_hud(water):
    baseline = with_bobber(water, 20, 20)  # A pre-existing red icon or another bobber.
    detector = BobberDetector()
    detector.begin([baseline] * 3)
    frame = with_bobber(baseline)
    assert detector.find(frame) is None
    found = find(detector, frame)
    assert found is not None
    assert abs(found.box.center[0] - 134) < 3


def test_ambiguous_and_transient_objects_never_selected(water):
    detector = BobberDetector()
    detector.begin([water] * 3)
    two = with_bobber(with_bobber(water), 220, 80)
    assert find(detector, two) is None
    assert detector.find(with_bobber(water)) is None
    assert detector.find(water) is None
    assert detector.find(with_bobber(water)) is None


def test_calibration_needs_similar_object_at_new_position(water):
    detector = BobberDetector()
    detector.begin([water] * 3)
    first = find(detector, with_bobber(water))
    calibration = Calibration()
    assert not calibration.observe(first)
    assert not calibration.observe(first)  # Fixed HUD position is insufficient.
    detector.begin([water] * 3)
    second = find(detector, with_bobber(water, 190, 110))
    assert calibration.observe(second)


def test_uniform_wave_does_not_trigger_bite(water):
    detector = BobberDetector()
    detector.begin([water] * 3)
    frame = with_bobber(water)
    found = find(detector, frame)
    bite = BiteDetector(frame, found)
    for i in range(30):
        changed = cv2.add(frame, np.full_like(frame, 35 if i % 2 else 0))
        assert not bite.update(changed, i * 0.05)[0]


def test_local_splash_triggers_once_warmed_up(water):
    detector = BobberDetector()
    detector.begin([water] * 3)
    frame = with_bobber(water)
    found = find(detector, frame)
    bite = BiteDetector(frame, found)
    for i in range(15):
        assert not bite.update(frame, i * 0.05)[0]
    box = found.box
    splash = frame.copy()
    cv2.rectangle(
        splash,
        (box.x - 4, box.y - 4),
        (box.x + box.width + 3, box.y + box.height + 3),
        (255, 255, 255),
        3,
    )
    assert not bite.update(splash, 0.8)[0]
    assert bite.update(frame, 0.85)[0]
