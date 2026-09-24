from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .model import DEFAULT_TUNING, Detection, Image, Rect, Tuning


def gray(image: Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def similarity(a: Image, b: Image) -> float:
    """Compare centered patches, rejecting featureless templates."""
    a = gray(cv2.resize(a, (32, 32))).astype(np.float32)
    b = gray(cv2.resize(b, (32, 32))).astype(np.float32)
    if min(a.std(), b.std()) < 3:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


class BobberDetector:
    """Novel, compact, persistent objects with a red/orange feather.

    The color prior is specific to the default 3.3.5a bobber; no downloaded
    template is needed. Rejects ambiguity instead of selecting the best of peers.
    """

    def __init__(self, tuning: Tuning = DEFAULT_TUNING):
        self.tuning = tuning
        self.baseline: Image | None = None
        self.noise: np.ndarray | None = None
        self.previous: Rect | None = None
        self.streak = 0

    def begin(self, frames: list[Image]) -> None:
        stack = np.stack([gray(f) for f in frames]).astype(np.float32)
        self.baseline = np.median(stack, axis=0).astype(np.uint8)
        self.noise = np.max(np.abs(stack - self.baseline), axis=0)
        self.previous = None
        self.streak = 0

    def find(self, frame: Image, template: Image | None = None) -> Detection | None:
        if self.baseline is None or self.baseline.shape != frame.shape[:2]:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Default feather; also admits orange under warm scene lighting.
        colored = (hsv[:, :, 0] < 28) | (hsv[:, :, 0] > 168)
        colored &= (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 85)
        diff = cv2.absdiff(gray(frame), self.baseline)
        novel = diff > np.maximum(24, self.noise * 2 + 12)
        mask = (colored & novel).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        h, w = frame.shape[:2]
        scale = max(0.6, h / 1080)
        candidates: list[Detection] = []
        for x, y, bw, bh, area in stats[1:count]:
            if not (5 * scale**2 <= area <= 1200 * scale**2):
                continue
            if max(bw, bh) > 65 * scale or not 0.12 <= bw / bh <= 5:
                continue
            pad = max(5, int(8 * scale))
            left, top = max(0, int(x) - pad), max(0, int(y) - pad)
            box = Rect(left, top, min(w - left, int(bw) + pad * 2), min(h - top, int(bh) + pad * 2))
            patch = box.crop(frame).copy()
            score = similarity(template, patch) if template is not None else 1.0
            if score >= self.tuning.match_threshold:
                candidates.append(Detection(box, score, patch))
        if len(candidates) != 1:
            self.streak = 0
            self.previous = None
            return None
        candidate = candidates[0]
        if self.previous is not None:
            dx = candidate.box.center[0] - self.previous.center[0]
            dy = candidate.box.center[1] - self.previous.center[1]
            self.streak = self.streak + 1 if dx * dx + dy * dy < (15 * scale) ** 2 else 1
        else:
            self.streak = 1
        self.previous = candidate.box
        return candidate if self.streak >= 4 else None


class Calibration:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.samples: list[Detection] = []
        self.template: Image | None = None

    def observe(self, detection: Detection) -> bool:
        for previous in self.samples:
            # Fixed HUD changes in the same place are not enough evidence.
            distance = np.linalg.norm(np.subtract(previous.box.center, detection.box.center))
            if (
                distance >= 8
                and similarity(previous.template, detection.template) >= self.threshold
            ):
                self.template = detection.template.copy()
                return True
        self.samples.append(detection)
        return False


class BiteDetector:
    """Local motion excess compared to the surrounding water and a rolling baseline."""

    def __init__(self, frame: Image, detection: Detection, floor: float = 0.07):
        self.box = detection.box
        self.template = detection.template
        self.previous = gray(frame)
        self.history: deque[float] = deque(maxlen=60)
        self.floor = floor
        self.spikes = 0
        self.last_score = 0.0
        self.last_seen = 0.0

    def update(self, frame: Image, elapsed: float) -> tuple[bool, bool]:
        current = gray(frame)
        x, y, bw, bh = self.box.x, self.box.y, self.box.width, self.box.height
        h, w = current.shape
        pad = max(bw, bh)
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(w, x + bw + pad), min(h, y + bh + pad)
        region = frame[top:bottom, left:right]
        response = cv2.matchTemplate(region, self.template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, location = cv2.minMaxLoc(response)
        if confidence >= 0.60:
            nx, ny = left + location[0], top + location[1]
            # Small movement is expected; do not follow unrelated distant matches.
            if abs(nx - x) <= 12 and abs(ny - y) <= 12:
                self.box = Rect(nx, ny, bw, bh)
                self.last_seen = elapsed
        changed = cv2.absdiff(current, self.previous) > 25
        self.previous = current
        inner = Rect(
            max(0, x - 5),
            max(0, y - 5),
            min(w - max(0, x - 5), bw + 10),
            min(h - max(0, y - 5), bh + 10),
        )
        local = float(inner.crop(changed).mean())
        ring = changed[top:bottom, left:right].copy()
        ring_mask = np.ones_like(ring, dtype=bool)
        ring_mask[
            inner.y - top : inner.y - top + inner.height,
            inner.x - left : inner.x - left + inner.width,
        ] = False
        background = float(ring[ring_mask].mean()) if ring_mask.any() else 0.0
        self.last_score = max(0.0, local - background * 1.5)
        ready = len(self.history) >= 12
        median = float(np.median(self.history)) if self.history else 0
        mad = float(np.median(np.abs(np.array(self.history) - median))) if self.history else 0
        threshold = max(self.floor, median + max(0.03, 6 * mad))
        spike = ready and self.last_score > threshold
        self.spikes = self.spikes + 1 if spike else 0
        if not spike:
            self.history.append(self.last_score)
        visible = elapsed - self.last_seen <= 0.5
        return self.spikes >= 2 and visible, visible
