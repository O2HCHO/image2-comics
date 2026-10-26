from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import tempfile

EDIT_MODE = "OpenAI 图像编辑（multipart）"
JSON_MODE = "自定义 Base64 JSON"
ORIGINAL = "保持原始比例"
RATIOS = {ORIGINAL: None, "1:1（透明补边）": (1, 1), "3:4（透明补边）": (3, 4), "16:9（透明补边）": (16, 9)}
DEFAULT_PROMPT = (
    "请为这张黑白漫画上色。严格保留原有线条、文字、对话框、分镜、人物姿态和构图，"
    "不改写或翻译文字，不增加、删除画面内容。仅添加自然协调的颜色，保留原稿细节与阴影。"
    "同一人物的头发、服装、肤色保持一致。"
)


def data_dir() -> Path:
    path = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "Image2Comics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".coloring-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


class Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def protect(value: bytes, decrypt: bool = False) -> bytes:
    if sys.platform != "win32":
        raise OSError("记住密钥功能仅支持 Windows DPAPI")
    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel.LocalFree(ctypes.cast(target.pbData, ctypes.c_void_p))


def load_settings() -> dict:
    location = data_dir() / "settings.json"
    if location.exists():
        settings = read_json(location)
    else:
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
        legacy = read_json(base / "local_settings.json")
        allowed = {"api_url", "model", "api_mode", "prompt", "denoising_strength", "output_ratio"}
        settings = {key: value for key, value in legacy.items() if key in allowed}
        settings["remember_api_key"] = False
    settings.pop("api_key", None)
    if settings.get("remember_api_key") and settings.get("protected_key"):
        try:
            settings["api_key"] = protect(base64.b64decode(settings["protected_key"]), decrypt=True).decode("utf-8")
        except (OSError, ValueError, UnicodeError):
            settings["key_error"] = "无法解密已保存的密钥，请重新输入。"
    return settings


def save_settings(settings: dict) -> None:
    result = dict(settings)
    key = result.pop("api_key", "")
    result.pop("protected_key", None)
    result.pop("key_error", None)
    if result.get("remember_api_key") and key:
        result["protected_key"] = base64.b64encode(protect(key.encode("utf-8"))).decode("ascii")
    atomic_json(data_dir() / "settings.json", result)
