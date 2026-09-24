from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .runtime import PreviewWorker, ReplayWorker, SessionWorker, session_name
from .settings import ALLOWED_KEYS, GAME_LANGUAGES, Settings, data_directory

# Harbor instruments: blue slate, sea glass, brass and a restrained red stop control.
STYLE = """
QMainWindow, QWidget#root { background: #172C38; color: #E8F0EF; }
QWidget { color: #E8F0EF; font-family: 'Segoe UI'; font-size: 13px; }
QLabel#title { font-family: Georgia; font-size: 32px; color: #E7C985; }
QLabel#muted { color: #A5BDC5; }
QLabel#status { font-size: 16px; font-weight: 600; color: #94D3C5; }
QFrame#panel { background: #203B49; border: 1px solid #355461; border-radius: 10px; }
QLabel#preview { background: #11232E; color: #A5BDC5; border-radius: 8px; }
QPushButton { background: #355461; border: 1px solid #526F7A; border-radius: 5px;
              padding: 6px 12px; min-height: 18px; }
QPushButton:hover { background: #466774; }
QPushButton:focus, QComboBox:focus, QLineEdit:focus { border: 2px solid #E7C985; }
QPushButton:disabled { color: #758D98; background: #263F4C; border-color: #355461; }
QPushButton#start { background: #94D3C5; color: #112E32; font-weight: 700; }
QPushButton#start:disabled { background: #355461; color: #758D98; }
QPushButton#stop { background: #AE5454; color: #FFFFFF; font-weight: 700; }
QPushButton#stop:disabled { background: #355461; color: #758D98; }
QLineEdit, QComboBox { background: #11232E; border: 1px solid #526F7A;
                     border-radius: 4px; padding: 7px; }
QComboBox QAbstractItemView { background: #203B49; selection-background-color: #355461; }
QPlainTextEdit { background: #11232E; border: 1px solid #355461; border-radius: 6px;
                 font-family: 'Cascadia Mono', monospace; font-size: 12px; }
QCheckBox { spacing: 8px; }
"""


class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__("Выберите окно WoW или откройте запись\nдля проверки распознавания")
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.original: QPixmap | None = None

    @Slot(object)
    def show_frame(self, frame) -> None:
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()
        self.original = QPixmap.fromImage(image)
        self._fit()

    def _fit(self):
        if self.original:
            self.setPixmap(
                self.original.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def clear_frame(self, message: str) -> None:
        self.original = None
        self.setText(message)


class MainWindow(QMainWindow):
    def __init__(self, storage: Path | None = None):
        super().__init__()
        self.storage = storage or data_directory()
        self.preview_worker = None
        self.worker = None
        self.pending_action = None
        self.closing = False
        self.windows = []
        self.video: Path | None = None
        self.setWindowTitle("Тихая вода · WoW Fishing")
        self.resize(1130, 800)
        self.setMinimumSize(880, 670)
        settings_error = ""
        try:
            self.settings = Settings.load(self.storage / "settings.json")
        except (ValueError, OSError, TypeError) as exc:
            self.settings = Settings()
            settings_error = f"Настройки сброшены: {exc}"
        self._build()
        self.refresh_windows()
        if settings_error:
            self.add_log(settings_error)
        self.stop_shortcut = QShortcut(QKeySequence("F8"), self)
        self.stop_shortcut.activated.connect(self.stop)

    def _build(self):
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(16)
        title = QLabel("Тихая вода", objectName="title")
        layout.addWidget(title)
        subtitle = QLabel(
            "Рыбалка в WoW 3.3.5a  /  автоматический поиск поплавка", objectName="muted"
        )
        layout.addWidget(subtitle)
        content = QHBoxLayout()
        content.setSpacing(18)
        panel = QFrame(objectName="panel")
        panel.setFixedWidth(316)
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(16, 18, 16, 18)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Окно игры"))
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self.selection_changed)
        controls.addWidget(self.window_combo)
        self.refresh_button = QPushButton("Обновить список окон")
        self.refresh_button.clicked.connect(self.refresh_windows)
        controls.addWidget(self.refresh_button)
        self.platform_hint = QLabel(
            "Оставьте выбранное окно видимым.\nПереключение в другое окно остановит бота."
            if sys.platform == "win32"
            else "На macOS доступна проверка записей.\nУправление игрой работает на Windows.",
            objectName="muted",
        )
        self.platform_hint.setWordWrap(True)
        controls.addWidget(self.platform_hint)
        form = QFormLayout()
        self.language_combo = QComboBox()
        for language, label in GAME_LANGUAGES.items():
            self.language_combo.addItem(label, language)
        self.language_combo.setCurrentIndex(
            self.language_combo.findData(self.settings.game_language)
        )
        self.language_combo.setToolTip("Выберите язык текста в игре или открытой записи")
        form.addRow("Язык клиента", self.language_combo)
        self.key_combo = QComboBox()
        self.key_combo.addItems([key.upper() for key in ALLOWED_KEYS])
        self.key_combo.setCurrentText(self.settings.cast_key.upper())
        form.addRow("Клавиша рыбалки", self.key_combo)
        controls.addLayout(form)
        controls.addWidget(QLabel("Tesseract — путь, если не найден"))
        self.tesseract = QLineEdit(self.settings.tesseract_path)
        self.tesseract.setPlaceholderText("Автоматически из PATH")
        controls.addWidget(self.tesseract)
        self.record = QCheckBox("Записывать сеанс для проверки")
        self.record.setChecked(self.settings.record)
        controls.addWidget(self.record)
        self.start_button = QPushButton("Начать рыбалку", objectName="start")
        self.start_button.clicked.connect(self.start_live)
        controls.addWidget(self.start_button)
        self.stop_button = QPushButton("Стоп  ·  F8", objectName="stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        self.open_button = QPushButton("Открыть запись…")
        self.open_button.clicked.connect(self.open_video)
        self.replay_ocr = QCheckBox("Проверять текст ошибок в записи")
        self.replay_ocr.setChecked(True)
        self.replay_button = QPushButton("Проверить запись")
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self.start_replay)
        content.addWidget(panel)
        right = QVBoxLayout()
        self.status = QLabel("Готово к настройке", objectName="status")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        self.preview = PreviewLabel()
        right.addWidget(self.preview, 1)
        self.preview_caption = QLabel("Изображение в границах выбранного окна", objectName="muted")
        self.preview_caption.setWordWrap(True)
        right.addWidget(self.preview_caption)
        self.count_label = QLabel("Забросы  0     /     Попытки сбора  0", objectName="muted")
        right.addWidget(self.count_label)
        playback = QHBoxLayout()
        playback.addWidget(self.open_button)
        playback.addWidget(self.replay_button)
        right.addLayout(playback)
        right.addWidget(self.replay_ocr)
        content.addLayout(right, 1)
        layout.addLayout(content, 1)
        layout.addWidget(QLabel("Журнал сеанса"))
        self.journal = QPlainTextEdit()
        self.journal.setReadOnly(True)
        self.journal.setMaximumBlockCount(500)
        self.journal.setFixedHeight(125)
        layout.addWidget(self.journal)

    @Slot(str)
    def add_log(self, text: str):
        self.journal.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    @Slot(int, int)
    def update_counts(self, casts: int, loots: int):
        self.count_label.setText(f"Забросы  {casts}     /     Попытки сбора  {loots}")

    def refresh_windows(self):
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        self.window_combo.addItem("Выберите клиент…", None)
        self.windows = []
        if sys.platform == "win32":
            from .windows import list_windows

            try:
                self.windows = list_windows()
            except Exception as exc:
                self.add_log(str(exc))
        for window in self.windows:
            self.window_combo.addItem(
                f"{window.title} · PID {window.pid} · {window.hwnd:X}", window
            )
        self.window_combo.blockSignals(False)
        self.window_combo.setEnabled(sys.platform == "win32")
        self.refresh_button.setEnabled(sys.platform == "win32")
        self.selection_changed()

    def selection_changed(self):
        self.start_button.setEnabled(self.window_combo.currentData() is not None)
        self._after_preview(self._preview_selected)

    def _after_preview(self, action):
        if self.preview_worker is not None:
            self.pending_action = action
            self.preview_worker.token.stop("Предпросмотр завершён")
        else:
            action()

    def _preview_finished(self):
        self.preview_worker.deleteLater()
        self.preview_worker = None
        action, self.pending_action = self.pending_action, None
        if self.closing:
            self.close()
        elif action:
            action()

    def _preview_selected(self):
        window = self.window_combo.currentData()
        if window is None or self.worker is not None or self.closing:
            self.preview.clear_frame("Выберите окно WoW или откройте запись")
            return
        self.preview_worker = PreviewWorker(window)
        self.preview_worker.frame.connect(self.preview.show_frame)
        self.preview_worker.failed.connect(self.add_log)
        self.preview_worker.finished.connect(self._preview_finished)
        self.preview_worker.start()

    def _read_settings(self) -> Settings:
        settings = Settings(
            cast_key=self.key_combo.currentText().lower(),
            tesseract_path=self.tesseract.text().strip(),
            record=self.record.isChecked(),
            game_language=self.language_combo.currentData(),
        )
        settings.save(self.storage / "settings.json")
        return settings

    def _busy(self, busy: bool):
        for widget in (
            self.window_combo,
            self.refresh_button,
            self.key_combo,
            self.language_combo,
            self.tesseract,
            self.record,
            self.open_button,
            self.replay_ocr,
        ):
            widget.setEnabled(not busy)
        if sys.platform != "win32":
            self.window_combo.setEnabled(False)
            self.refresh_button.setEnabled(False)
        self.start_button.setEnabled(not busy and self.window_combo.currentData() is not None)
        self.replay_button.setEnabled(not busy and self.video is not None)
        self.stop_button.setEnabled(busy)

    def _launch(self, worker):
        self.worker = worker
        worker.frame.connect(self.preview.show_frame)
        worker.log.connect(self.add_log)
        worker.status.connect(self.status.setText)
        worker.counts.connect(self.update_counts)
        worker.finished.connect(self._finished)
        worker.start()

    def start_live(self):
        window = self.window_combo.currentData()
        if window is None or self.worker is not None:
            return
        try:
            settings = self._read_settings()
        except OSError as exc:
            self.add_log(f"Не удалось сохранить настройки: {exc}")
            return
        self._busy(True)
        self.preview_caption.setText(
            f"{window.title} · PID {window.pid} · F8 останавливает рыбалку"
        )
        self._after_preview(
            lambda: self._launch(
                SessionWorker(window, settings, self.storage / "sessions" / session_name())
            )
        )

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Открыть запись клиента", "", "Видео (*.avi *.mp4 *.mkv *.mov)"
        )
        if filename:
            self.video = Path(filename)
            self.replay_button.setEnabled(True)
            self.preview_caption.setText(
                f"Запись: {self.video.name}. Ввод будет только имитироваться."
            )
            self.add_log(f"Выбрана запись: {self.video}")

    def start_replay(self):
        if self.video is None or self.worker is not None:
            return
        try:
            settings = self._read_settings()
        except OSError as exc:
            self.add_log(str(exc))
            return
        self._busy(True)
        self.status.setText("Проверка записи без управления игрой…")
        self._after_preview(
            lambda: self._launch(
                ReplayWorker(
                    self.video,
                    settings,
                    self.storage / "reports" / f"{session_name()}.json",
                    self.replay_ocr.isChecked(),
                )
            )
        )

    def stop(self):
        self.pending_action = None
        if self.worker is not None:
            self.worker.token.stop("Остановлено пользователем")
        else:
            self._busy(False)

    def _finished(self):
        self.worker.deleteLater()
        self.worker = None
        self._busy(False)
        # Do not automatically resume capture or input after loss of focus.
        if self.closing:
            self.close()

    def closeEvent(self, event: QCloseEvent):
        self.closing = True
        self.stop()
        if self.preview_worker:
            self.preview_worker.token.stop("Приложение закрывается")
        if self.worker or self.preview_worker:
            event.ignore()
            self.status.setText("Завершение сеанса…")
            return
        event.accept()


def main() -> None:
    if sys.platform == "win32":
        from .windows import enable_dpi_awareness

        enable_dpi_awareness()
    app = QApplication(sys.argv)
    app.setApplicationName("WoWFishing")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    # Allows a noninteractive render/import check without any live capture.
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(250, window.close)
    sys.exit(app.exec())
