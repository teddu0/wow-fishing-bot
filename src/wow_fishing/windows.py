"""Windows-only adapters. Import lazily so analysis and GUI work on other platforms."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import PureWindowsPath

from .model import Image, Rect, Window
from .safety import GuardedInput, StopToken


def enable_dpi_awareness() -> None:
    # Must run before QApplication, pynput or dxcam; use physical pixels everywhere.
    user32 = ctypes.windll.user32
    user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


def process_info(pid: int) -> tuple[str, float]:
    import win32api
    import win32process

    handle = win32api.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query_name = kernel32.QueryFullProcessImageNameW
        query_name.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        query_name.restype = wintypes.BOOL
        length = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(length.value)
        if not query_name(int(handle), 0, buffer, ctypes.byref(length)):
            raise ctypes.WinError(ctypes.get_last_error())
        path = buffer.value
        started = win32process.GetProcessTimes(handle)["CreationTime"].timestamp()
        return PureWindowsPath(path).name.lower(), started
    finally:
        handle.Close()


def list_windows() -> list[Window]:
    import win32gui
    import win32process

    found = []

    def visit(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            name, started = process_info(pid)
        except Exception:
            return
        if name in {"wow.exe", "wow-64.exe"} or win32gui.GetClassName(hwnd) == "GxWindowClass":
            found.append(Window(hwnd, pid, title or "World of Warcraft", started))

    win32gui.EnumWindows(visit, None)
    return found


def client_rect(window: Window) -> Rect:
    import win32gui

    left, top, right, bottom = win32gui.GetClientRect(window.hwnd)
    x, y = win32gui.ClientToScreen(window.hwnd, (left, top))
    return Rect(x, y, right - left, bottom - top)


class WindowGuard:
    def __init__(self, window: Window):
        self.window = window
        self.rect = client_rect(window)
        self.validate(focus=False)

    def validate(self, focus: bool = True) -> None:
        import win32gui
        import win32process

        hwnd = self.window.hwnd
        if not win32gui.IsWindow(hwnd):
            raise RuntimeError("Выбранное окно закрыто — выберите клиент заново")
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid != self.window.pid or process_info(pid)[1] != self.window.process_started:
            raise RuntimeError("Выбранный процесс изменился — выберите клиент заново")
        if win32gui.IsIconic(hwnd) or not win32gui.IsWindowVisible(hwnd):
            raise RuntimeError("Выбранное окно свёрнуто или скрыто")
        if client_rect(self.window) != self.rect or self.rect.width < 100 or self.rect.height < 100:
            raise RuntimeError("Геометрия окна изменилась — запустите калибровку заново")
        if focus and win32gui.GetForegroundWindow() != hwnd:
            raise RuntimeError("Выбранное окно потеряло фокус")


def monitor_region(rect: Rect, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    if not (
        left <= rect.x
        and top <= rect.y
        and rect.x + rect.width <= right
        and rect.y + rect.height <= bottom
    ):
        raise RuntimeError("Разместите окно целиком на одном мониторе")
    return rect.x - left, rect.y - top, rect.x + rect.width - left, rect.y + rect.height - top


class DesktopSource:
    def __init__(self, guard: WindowGuard):
        import dxcam
        import win32api

        monitor = win32api.MonitorFromWindow(guard.window.hwnd, 2)
        info = win32api.GetMonitorInfo(monitor)
        self.region = monitor_region(guard.rect, info["Monitor"])
        self.camera = None
        # DXcam 0.3.0 exposes monitor names only on its factory outputs. Keep this
        # version-specific bridge here; never assume Win32 and DXGI indices match.
        factory = getattr(dxcam, "__factory")
        for device_idx, outputs in enumerate(factory.outputs):
            for output_idx, output in enumerate(outputs):
                if output.devicename == info["Device"]:
                    self.camera = dxcam.create(
                        device_idx=device_idx, output_idx=output_idx, output_color="BGR"
                    )
                    break
            if self.camera is not None:
                break
        if self.camera is None:
            raise RuntimeError("DXcam не нашёл монитор выбранного окна")

    def read(self) -> Image | None:
        frame = self.camera.grab(region=self.region, new_frame_only=False)
        return frame.copy() if frame is not None else None

    def close(self) -> None:
        self.camera.release()


def make_input(guard: WindowGuard, token: StopToken, key_name: str) -> GuardedInput:
    from pynput import keyboard, mouse

    key = (
        getattr(keyboard.Key, key_name)
        if key_name.startswith("f") and len(key_name) > 1
        else key_name
    )
    return GuardedInput(
        token,
        guard.validate,
        keyboard.Controller(),
        mouse.Controller(),
        key,
        mouse.Button.right,
        (guard.rect.x, guard.rect.y),
    )


def start_hotkey(token: StopToken):
    from pynput import keyboard

    def pressed(key):
        if key == keyboard.Key.f8:
            token.stop("Остановлено горячей клавишей F8")

    listener = keyboard.Listener(on_press=pressed)
    listener.start()
    listener.wait()
    return listener
