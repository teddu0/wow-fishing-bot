from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from .engine import FishingEngine
from .model import Image, OcrResult, State
from .ocr import ErrorReader, check_tesseract
from .recording import RecordingInput, VideoSource
from .safety import StopToken
from .settings import GAME_LANGUAGES


def replay(
    video: Path,
    *,
    use_ocr: bool = True,
    tesseract_path: str = "",
    game_language: str = "rus",
    timeline: Path | None = None,
    token: StopToken | None = None,
    on_frame: Callable[[Image, FishingEngine], None] = lambda *_: None,
    emit: Callable[[str], None] = lambda _: None,
) -> dict:
    """Run the same state machine with virtual time and a recording input adapter.

    A replay cannot change a pre-existing video. Cast times must align with the
    recorded session; action timestamps in the report expose alignment errors.
    """
    reader = ErrorReader(game_language)
    if use_ocr:
        check_tesseract(tesseract_path, game_language)
    source = VideoSource(video)
    inputs = RecordingInput()
    engine = None
    times = None
    if timeline is None:
        sibling = video.with_name("frames.jsonl")
        timeline = sibling if sibling.exists() else None
    try:
        if timeline:
            times = [json.loads(line)["time"] for line in timeline.read_text().splitlines()]
            if any(b <= a for a, b in zip(times, times[1:], strict=False)):
                raise ValueError("Времена кадров должны строго возрастать")
        last_ocr = float("-inf")
        sequence = 0
        while token is None or not token.event.is_set():
            frame = source.read()
            if frame is None:
                break
            if times is not None and source.index >= len(times):
                raise ValueError("В журнале времени не хватает кадров")
            now = times[source.index] if times is not None else source.index / source.fps
            if engine is None:
                engine = FishingEngine(inputs, now, emit)
            inputs.now = now
            if now - last_ocr >= 0.4:
                text, full = reader.read(frame) if use_ocr else ("", False)
                engine.accept_ocr(OcrResult(sequence, now, text, full))
                sequence += 1
                last_ocr = now
            engine.tick(frame, now)
            on_frame(frame, engine)
            if engine.state == State.STOPPED:
                break
        if engine is None:
            raise ValueError("В записи нет кадров")
        return {
            "mode": "replay-no-input",
            "ocr_enabled": use_ocr,
            "game_language": game_language,
            "frames": source.index + 1,
            "calibrated": engine.calibration.template is not None,
            "state": engine.state.value,
            "reason": token.reason if token and token.event.is_set() else engine.reason,
            "actions": inputs.actions,
            "note": "Команды имитированы; успешный сбор добычи не подтверждается записью команд.",
        }
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка записи без управления мышью и клавиатурой"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--report", type=Path, default=Path("replay-report.json"))
    parser.add_argument("--tesseract", default="")
    parser.add_argument(
        "--game-language",
        choices=GAME_LANGUAGES,
        default="rus",
        help="Язык клиента в записи: rus — русский (по умолчанию), eng — английский",
    )
    parser.add_argument("--without-ocr", action="store_true", help="Проверять только зрение и цикл")
    args = parser.parse_args()
    try:
        result = replay(
            args.video,
            timeline=args.timeline,
            use_ocr=not args.without_ocr,
            tesseract_path=args.tesseract,
            game_language=args.game_language,
            emit=print,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {args.report}")


if __name__ == "__main__":
    main()
