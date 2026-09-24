from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def data_directory() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "WoWFishing"


ALLOWED_KEYS = tuple("1234567890") + tuple(f"f{i}" for i in range(1, 13) if i != 8)
GAME_LANGUAGES = {"rus": "Русский", "eng": "Английский"}


def validate_game_language(language: str) -> None:
    if not isinstance(language, str) or language not in GAME_LANGUAGES:
        raise ValueError("Язык клиента должен быть русским (rus) или английским (eng)")


@dataclass
class Settings:
    cast_key: str = "1"
    tesseract_path: str = ""
    record: bool = False
    game_language: str = "rus"

    def __post_init__(self) -> None:
        validate_game_language(self.game_language)

    @classmethod
    def load(cls, path: Path) -> Settings:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Некорректный файл настроек")
        settings = cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
        if settings.cast_key not in ALLOWED_KEYS:
            raise ValueError("Клавиша заброса должна быть 0–9 или F1–F12, кроме F8")
        if not isinstance(settings.tesseract_path, str) or not isinstance(settings.record, bool):
            raise ValueError("Некорректные типы настроек")
        return settings

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
