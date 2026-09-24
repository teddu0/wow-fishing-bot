"""Optional real OCR test; unit tests remain usable without a Tesseract installation."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from wow_fishing.ocr import ErrorReader, check_tesseract
from wow_fishing.recording import Recorder
from wow_fishing.replay import replay


@pytest.fixture
def text_frame():
    fonts = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((path for path in fonts if path.exists()), None)
    if font_path is None:
        pytest.skip("No Cyrillic/Latin font available for the synthetic fixture")

    def render(phrase, font_size=24):
        font = ImageFont.truetype(str(font_path), font_size)
        image = Image.new("RGB", (900, 600), (35, 65, 80))
        ImageDraw.Draw(image).text((190, 110), phrase, fill=(255, 30, 30), font=font)
        return np.array(image)[:, :, ::-1].copy()

    return render


@pytest.fixture
def tesseract_command(language):
    command = shutil.which("tesseract")
    if command is None:
        pytest.skip("Tesseract is not installed")
    result = subprocess.run(
        [command, "--list-langs"], capture_output=True, text=True, timeout=4, check=True
    )
    if language not in result.stdout.splitlines():
        pytest.skip(f"{language}.traineddata is not installed")
    return command


@pytest.mark.parametrize(
    ("language", "phrase", "full"),
    [
        ("rus", "Инвентарь заполнен.", True),
        ("rus", "Ваши сумки заполнены.", True),
        ("rus", "Нет места в сумках.", True),
        ("rus", "Недостаточно маны.", False),
        ("eng", "Inventory is full.", True),
        ("eng", "Your bags are full.", True),
        ("eng", "Not enough room in your bags.", True),
        ("eng", "Not enough mana.", False),
        ("eng", "Your bank is full.", False),
    ],
)
def test_text_through_real_tesseract(language, phrase, full, text_frame, tesseract_command):
    check_tesseract(tesseract_command, language)
    text, detected = ErrorReader(language).read(text_frame(phrase))
    assert text, "OCR must read the negative examples as well"
    assert detected == full, text


@pytest.mark.parametrize(
    ("language", "phrase"), [("rus", "Инвентарь заполнен."), ("eng", "Inventory is full.")]
)
def test_full_bags_video_stops_before_cast(
    tmp_path, language, phrase, text_frame, tesseract_command
):
    # Use a larger font for lossy MJPEG; the image tests also cover smaller text.
    frame = text_frame(phrase, font_size=32)
    recorder = Recorder(tmp_path, frame.shape)
    try:
        for index in range(30):
            recorder.write(frame, index * 0.1)
    finally:
        recorder.close()
    result = replay(
        tmp_path / "screen.avi", tesseract_path=tesseract_command, game_language=language
    )
    assert result["game_language"] == language
    assert result["state"] == "Остановлено"
    assert "Сумки заполнены" in result["reason"]
    assert result["actions"] == []
