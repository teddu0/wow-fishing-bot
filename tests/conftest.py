import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def water():
    return np.full((240, 320, 3), (80, 65, 35), np.uint8)


def with_bobber(frame, x=130, y=90):
    image = frame.copy()
    cv2.rectangle(image, (x, y), (x + 8, y + 13), (30, 40, 240), -1)
    cv2.rectangle(image, (x + 3, y + 2), (x + 4, y + 8), (220, 230, 240), -1)
    return image
