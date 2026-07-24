"""Image-to-image batch colouring desktop application.

The program sends one input image at a time to a compatible image-to-image
HTTP API.  Update build_payload() / extract_image_base64() if the selected API
uses a different request or response schema.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import queue
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import customtkinter as ctk
import requests
from PIL import Image, UnidentifiedImageError
from tkinter import filedialog, messagebox


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
REQUEST_TIMEOUT_SECONDS = 300
RATE_LIMIT_DELAY_SECONDS = 2
OPENAI_IMAGE_EDIT_MODE = "OpenAI 图像编辑（multipart）"
CUSTOM_JSON_MODE = "自定义 Base64 JSON"
KEEP_ORIGINAL_RATIO = "保持原始比例"
RATIO_OPTIONS = {
    KEEP_ORIGINAL_RATIO: None,
    "1:1（透明补边）": (1, 1),
    "3:4（透明补边）": (3, 4),
    "16:9（透明补边）": (16, 9),
}
SETTINGS_FILE = Path(__file__).with_name("local_settings.json")


def natural_sort_key(path: Path) -> list[Any]:
    """Sort names such as page_2.png before page_10.png."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def remove_data_url_prefix(value: str) -> str:
    """Accept both raw Base64 and data:image/...;base64,... values."""
    if value.startswith("data:"):
        try:
            return value.split(",", 1)[1]
        except IndexError as error:
            raise ValueError("返回的 Data URL 不包含 Base64 内容") from error
    return value


def build_payload(image_base64: str, prompt: str, denoising_strength: float, model: str) -> dict[str, Any]:
    """Build the request body expected by the target image-to-image endpoint."""
    return {
        "model": model,
        "image": image_base64,
        "prompt": prompt,
        "denoising_strength": denoising_strength,
    }


def add_transparent_padding(image_bytes: bytes, ratio_option: str) -> bytes:
    """Expand a PNG canvas to a selected aspect ratio without cropping or scaling."""
    ratio = RATIO_OPTIONS.get(ratio_option)
    if ratio is None:
        return image_bytes

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            image = source.convert("RGBA")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("无法读取 API 返回的图像，不能应用输出比例") from error

    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("API 返回的图像尺寸无效")
    ratio_width, ratio_height = ratio
    if width * ratio_height < height * ratio_width:
        multiplier = (height + ratio_height - 1) // ratio_height
    else:
        multiplier = (width + ratio_width - 1) // ratio_width
    canvas_width = multiplier * ratio_width
    canvas_height = multiplier * ratio_height

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    offset = ((canvas_width - width) // 2, (canvas_height - height) // 2)
    canvas.alpha_composite(image, offset)
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def load_local_settings() -> dict[str, Any]:
    """Load non-portable UI preferences; invalid local files are ignored."""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_local_settings(data: dict[str, Any]) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_response(response: requests.Response) -> Any:
    """Return JSON, with a useful error when a documentation/web page was called."""
    content_type = response.headers.get("Content-Type", "").casefold()
    if "html" in content_type:
        raise ValueError(
            "API 返回了 HTML 网页而不是 JSON。请检查 API URL：它很可能是文档/官网地址，"
            f"不是实际的图生图接口。状态码: {response.status_code}，最终 URL: {response.url}"
        )
    try:
        return response.json()
    except (json.JSONDecodeError, requests.JSONDecodeError) as error:
        preview = response.text[:200].replace("\n", " ")
        raise ValueError(
            f"API 返回内容不是有效 JSON。状态码: {response.status_code}，最终 URL: {response.url}，"
            f"Content-Type: {content_type or '未提供'}，内容片段: {preview!r}"
        ) from error


def api_error_detail(response: requests.Response) -> str:
    """Extract a useful error message from common OpenAI-compatible responses."""
    try:
        payload = response.json()
    except (json.JSONDecodeError, requests.JSONDecodeError):
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or error.get("type")
            if message:
                return str(message)
        if isinstance(error, str) and error.strip():
            return error.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    preview = response.text[:300].replace("\n", " ").strip()
    return preview or "服务端未返回错误详情"


def model_endpoint_candidates(api_url: str) -> list[str]:
    """Derive safe, same-host OpenAI-compatible model-list endpoints."""
    parsed = urlsplit(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API URL 必须是有效的 http:// 或 https:// 地址")

    path = parsed.path.rstrip("/")
    match = re.search(r"/v1(?:/|$)", path, flags=re.IGNORECASE)
    if match:
        api_base_path = path[: match.end()].rstrip("/")
        paths = [f"{api_base_path}/models", "/models"]
    else:
        paths = ["/v1/models", "/models"]

    candidates: list[str] = []
    for candidate_path in paths:
        candidate = urlunsplit((parsed.scheme, parsed.netloc, candidate_path, "", ""))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_model_ids(data: Any) -> list[str]:
    """Read OpenAI's {data: [{id: ...}]} response, plus common alternatives."""
    if not isinstance(data, dict):
        raise ValueError("模型列表响应不是 JSON 对象")
    raw_models = data.get("data") or data.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("模型列表响应中未找到 data 数组")

    model_ids: list[str] = []
    for item in raw_models:
        model_id = item.get("id") if isinstance(item, dict) else item
        if isinstance(model_id, str) and model_id.strip() and model_id not in model_ids:
            model_ids.append(model_id)
    if not model_ids:
        raise ValueError("模型列表为空或不包含 id 字段")
    return model_ids


def extract_image_base64(data: Any) -> str:
    """Extract the first image from common image-generation API response shapes."""
    if not isinstance(data, dict):
        raise ValueError("API 返回的 JSON 根节点不是对象")

    image: Any = data.get("image")
    if not image and isinstance(data.get("images"), list) and data["images"]:
        image = data["images"][0]
    if not image and isinstance(data.get("data"), list) and data["data"]:
        image = data["data"][0]
    if not image and isinstance(data.get("data"), dict):
        nested = data["data"]
        image = nested.get("image")
        if not image and isinstance(nested.get("images"), list) and nested["images"]:
            image = nested["images"][0]

    if isinstance(image, dict):
        image = image.get("b64_json") or image.get("image")
    if not isinstance(image, str) or not image.strip():
        raise ValueError("API 返回 JSON 中未找到 image、images[0] 或 data.image 图像字段")
    return remove_data_url_prefix(image.strip())


@dataclass(frozen=True)
class JobConfig:
    input_dir: Path
    output_dir: Path
    selected_files: tuple[Path, ...]
    api_url: str
    api_key: str
    model: str
    api_mode: str
    prompt: str
    denoising_strength: float
    output_ratio: str


class BatchWorker(threading.Thread):
    """A worker that never accesses tkinter objects directly."""

    def __init__(self, config: JobConfig, stop_event: threading.Event, events: queue.Queue[tuple[str, Any]]) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.events = events

    def emit(self, event_type: str, payload: Any = None) -> None:
        self.events.put((event_type, payload))

    def run(self) -> None:
        try:
            files = sorted(
                (
                    item
                    for item in self.config.selected_files
                    if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
                ),
                key=natural_sort_key,
            )
        except OSError as error:
            self.emit("log", ("error", f"无法读取已选择的图片: {error}"))
            self.emit("finished", False)
            return

        if not files:
            self.emit("log", ("error", "输入文件夹中没有 PNG、JPG 或 JPEG 图片。"))
            self.emit("finished", False)
            return

        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.emit("log", ("error", f"无法创建输出文件夹: {error}"))
            self.emit("finished", False)
            return

        self.emit("set_total", len(files))
        self.emit("log", ("info", f"发现 {len(files)} 张图片，开始处理。"))
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

        completed = 0
        stopped = False
        for source_path in files:
            if self.stop_event.is_set():
                stopped = True
                break

            self.emit("log", ("info", f"正在处理: {source_path.name}"))
            try:
                if self.config.api_mode == OPENAI_IMAGE_EDIT_MODE:
                    mime_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
                    with source_path.open("rb") as image_file:
                        response = requests.post(
                            self.config.api_url,
                            headers={"Authorization": f"Bearer {self.config.api_key}"},
                            data={
                                "model": self.config.model,
                                "prompt": self.config.prompt,
                                "n": "1",
                                "output_format": "png",
                            },
                            files={"image": (source_path.name, image_file, mime_type)},
                            timeout=REQUEST_TIMEOUT_SECONDS,
                        )
                    output_name = f"colored_{source_path.stem}.png"
                else:
                    with source_path.open("rb") as image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode("ascii")
                    response = requests.post(
                        self.config.api_url,
                        headers=headers,
                        json=build_payload(image_base64, self.config.prompt, self.config.denoising_strength, self.config.model),
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    output_name = f"colored_{source_path.name}"
                if not response.ok:
                    raise ValueError(
                        f"HTTP {response.status_code}，最终 URL: {response.url}，服务端信息: {api_error_detail(response)}"
                    )
                response_data = parse_json_response(response)

                output_base64 = extract_image_base64(response_data)
                try:
                    output_bytes = base64.b64decode(output_base64, validate=True)
                except (ValueError, base64.binascii.Error) as error:
                    raise ValueError("API 返回的图像字段不是有效的 Base64 数据") from error
                if not output_bytes:
                    raise ValueError("API 返回的图像数据为空")

                output_bytes = add_transparent_padding(output_bytes, self.config.output_ratio)
                if self.config.output_ratio != KEEP_ORIGINAL_RATIO:
                    output_name = f"colored_{source_path.stem}.png"
                output_path = self.config.output_dir / output_name
                output_path.write_bytes(output_bytes)
                self.emit("log", ("success", f"完成: {source_path.name} -> {output_path.name}"))
            except requests.Timeout:
                self.emit("log", ("error", f"超时，已跳过: {source_path.name}"))
            except requests.RequestException as error:
                self.emit("log", ("error", f"请求失败，已跳过 {source_path.name}: {error}"))
            except (OSError, ValueError) as error:
                self.emit("log", ("error", f"处理失败，已跳过 {source_path.name}: {error}"))
            except Exception as error:  # Prevent one unexpected file/API issue from stopping the batch.
                self.emit("log", ("error", f"未知错误，已跳过 {source_path.name}: {error}"))

            completed += 1
            self.emit("progress", completed)
            if completed < len(files) and self.stop_event.wait(RATE_LIMIT_DELAY_SECONDS):
                stopped = True
                break

        if stopped:
            self.emit("log", ("warning", "已安全停止，未处理的图片保留在输入文件夹。"))
        else:
            self.emit("log", ("success", "批量处理完成。"))
        self.emit("finished", stopped)


class ConnectionWorker(threading.Thread):
    """Fetch available models without blocking the GUI thread."""

    def __init__(self, api_url: str, api_key: str, events: queue.Queue[tuple[str, Any]]) -> None:
        super().__init__(daemon=True)
        self.api_url = api_url
        self.api_key = api_key
        self.events = events

    def run(self) -> None:
        errors: list[str] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            candidates = model_endpoint_candidates(self.api_url)
        except ValueError as error:
            self.events.put(("models_result", (False, str(error), [])))
            return

        for endpoint in candidates:
            try:
                response = requests.get(endpoint, headers=headers, timeout=20)
            except requests.RequestException as error:
                errors.append(f"{endpoint}: 连接失败: {error}")
                continue

            if not response.ok:
                errors.append(f"{endpoint}: HTTP {response.status_code}: {api_error_detail(response)}")
                continue
            try:
                model_ids = extract_model_ids(parse_json_response(response))
            except ValueError as error:
                errors.append(f"{endpoint}: {error}")
                continue

            self.events.put(("models_result", (True, f"连接成功: {endpoint}，获取到 {len(model_ids)} 个模型。", model_ids)))
            return

        details = "\n".join(errors) if errors else "未找到可用的模型列表接口。"
        self.events.put(("models_result", (False, f"模型列表获取失败:\n{details}", [])))


class ImageColoringApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("图生图批量上色")
        self.geometry("920x830")
        self.minsize(760, 680)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.content = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.content.grid(row=0, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)

        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: BatchWorker | None = None
        self.connection_worker: ConnectionWorker | None = None
        self.total_files = 0
        self.local_settings = load_local_settings()
        self.available_images: list[Path] = []
        self.image_selection_vars: dict[Path, ctk.BooleanVar] = {}

        default_input = Path(__file__).with_name("inputs")
        default_output = Path(__file__).with_name("outputs")
        self.input_path_var = ctk.StringVar(value=str(default_input) if default_input.is_dir() else "")
        self.output_path_var = ctk.StringVar(value=str(default_output) if default_output.is_dir() else "")
        self.api_url_var = ctk.StringVar(value=str(self.local_settings.get("api_url", "")))
        self.api_key_var = ctk.StringVar(value=str(self.local_settings.get("api_key", "")))
        self.remember_api_key_var = ctk.BooleanVar(value=bool(self.local_settings.get("remember_api_key", False)))
        self.model_var = ctk.StringVar(value=str(self.local_settings.get("model", "请先测试连接")))
        saved_mode = self.local_settings.get("api_mode", OPENAI_IMAGE_EDIT_MODE)
        self.api_mode_var = ctk.StringVar(value=saved_mode if saved_mode in {OPENAI_IMAGE_EDIT_MODE, CUSTOM_JSON_MODE} else OPENAI_IMAGE_EDIT_MODE)
        saved_ratio = self.local_settings.get("output_ratio", KEEP_ORIGINAL_RATIO)
        self.output_ratio_var = ctk.StringVar(value=saved_ratio if saved_ratio in RATIO_OPTIONS else KEEP_ORIGINAL_RATIO)
        try:
            saved_denoise = float(self.local_settings.get("denoising_strength", 0.55))
        except (TypeError, ValueError):
            saved_denoise = 0.55
        self.denoise_var = ctk.DoubleVar(value=min(1.0, max(0.0, saved_denoise)))
        self.saved_prompt = str(self.local_settings.get("prompt", ""))

        self._build_ui()
        self._refresh_image_list(select_all=True)
        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        paths = ctk.CTkFrame(self.content)
        paths.grid(row=0, column=0, padx=18, pady=(18, 8), sticky="ew")
        paths.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(paths, text="文件夹", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w")
        ctk.CTkButton(paths, text="选择输入文件夹", command=self._choose_input).grid(row=1, column=0, padx=(14, 8), pady=6)
        self.input_entry = ctk.CTkEntry(paths, textvariable=self.input_path_var, state="readonly")
        self.input_entry.grid(row=1, column=1, padx=(0, 14), pady=6, sticky="ew")
        ctk.CTkButton(paths, text="选择输出文件夹", command=self._choose_output).grid(row=2, column=0, padx=(14, 8), pady=(6, 12))
        self.output_entry = ctk.CTkEntry(paths, textvariable=self.output_path_var, state="readonly")
        self.output_entry.grid(row=2, column=1, padx=(0, 14), pady=(6, 12), sticky="ew")

        selection = ctk.CTkFrame(self.content)
        selection.grid(row=1, column=0, padx=18, pady=8, sticky="ew")
        selection.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(selection, text="待处理图片", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(12, 6), sticky="w"
        )
        actions = ctk.CTkFrame(selection, fg_color="transparent")
        actions.grid(row=0, column=1, padx=14, pady=(12, 6), sticky="e")
        ctk.CTkButton(actions, text="全选", width=68, command=self._select_all_images).grid(row=0, column=0, padx=(0, 6))
        ctk.CTkButton(actions, text="清空", width=68, command=self._clear_image_selection).grid(row=0, column=1, padx=(0, 6))
        ctk.CTkButton(actions, text="刷新列表", width=88, command=self._refresh_image_list).grid(row=0, column=2)
        self.selection_count_label = ctk.CTkLabel(selection, text="已选择 0/0 张")
        self.selection_count_label.grid(row=1, column=0, columnspan=2, padx=14, pady=(0, 6), sticky="w")
        self.image_list_frame = ctk.CTkScrollableFrame(selection, height=170)
        self.image_list_frame.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 14), sticky="ew")
        self.image_list_frame.grid_columnconfigure(0, weight=1)

        settings = ctk.CTkFrame(self.content)
        settings.grid(row=2, column=0, padx=18, pady=8, sticky="ew")
        settings.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(settings, text="接口参数", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w")
        ctk.CTkLabel(settings, text="API URL").grid(row=1, column=0, padx=(14, 8), pady=6, sticky="w")
        url_frame = ctk.CTkFrame(settings, fg_color="transparent")
        url_frame.grid(row=1, column=1, padx=(0, 14), pady=6, sticky="ew")
        url_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(
            url_frame,
            textvariable=self.api_url_var,
            placeholder_text="https://api.example.com/v1/images/edits",
        ).grid(row=0, column=0, sticky="ew")
        self.test_connection_button = ctk.CTkButton(
            url_frame,
            text="测试连接并获取模型",
            width=150,
            command=self._test_connection,
        )
        self.test_connection_button.grid(row=0, column=1, padx=(8, 0))
        ctk.CTkLabel(settings, text="API Key").grid(row=2, column=0, padx=(14, 8), pady=6, sticky="w")
        key_frame = ctk.CTkFrame(settings, fg_color="transparent")
        key_frame.grid(row=2, column=1, padx=(0, 14), pady=6, sticky="ew")
        key_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(key_frame, textvariable=self.api_key_var, show="*").grid(row=0, column=0, sticky="ew")
        self.remember_api_key_checkbox = ctk.CTkCheckBox(
            key_frame,
            text="保存 API Key（本机明文）",
            variable=self.remember_api_key_var,
        )
        self.remember_api_key_checkbox.grid(row=0, column=1, padx=(8, 0))
        ctk.CTkLabel(settings, text="接口模式").grid(row=3, column=0, padx=(14, 8), pady=6, sticky="w")
        self.api_mode_combobox = ctk.CTkComboBox(
            settings,
            variable=self.api_mode_var,
            values=[OPENAI_IMAGE_EDIT_MODE, CUSTOM_JSON_MODE],
            state="readonly",
        )
        self.api_mode_combobox.grid(row=3, column=1, padx=(0, 14), pady=6, sticky="ew")
        ctk.CTkLabel(settings, text="模型").grid(row=4, column=0, padx=(14, 8), pady=6, sticky="w")
        self.model_combobox = ctk.CTkComboBox(
            settings,
            variable=self.model_var,
            values=[self.model_var.get()] if self.model_var.get() != "请先测试连接" else ["请先测试连接"],
            state="readonly",
        )
        self.model_combobox.grid(row=4, column=1, padx=(0, 14), pady=6, sticky="ew")
        ctk.CTkLabel(settings, text="Prompt").grid(row=5, column=0, padx=(14, 8), pady=6, sticky="nw")
        self.prompt_textbox = ctk.CTkTextbox(settings, height=100)
        self.prompt_textbox.grid(row=5, column=1, padx=(0, 14), pady=6, sticky="ew")
        if self.saved_prompt:
            self.prompt_textbox.insert("1.0", self.saved_prompt)
        ctk.CTkLabel(settings, text="重绘幅度").grid(row=6, column=0, padx=(14, 8), pady=(6, 12), sticky="w")
        denoise_frame = ctk.CTkFrame(settings, fg_color="transparent")
        denoise_frame.grid(row=6, column=1, padx=(0, 14), pady=(6, 12), sticky="ew")
        denoise_frame.grid_columnconfigure(0, weight=1)
        self.denoise_slider = ctk.CTkSlider(denoise_frame, from_=0, to=1, number_of_steps=100, variable=self.denoise_var, command=self._update_denoise_label)
        self.denoise_slider.grid(row=0, column=0, padx=(0, 12), sticky="ew")
        self.denoise_label = ctk.CTkLabel(denoise_frame, text="0.55", width=42)
        self.denoise_label.grid(row=0, column=1)
        self._update_denoise_label(self.denoise_var.get())
        ctk.CTkLabel(settings, text="输出比例").grid(row=7, column=0, padx=(14, 8), pady=(0, 12), sticky="w")
        self.output_ratio_combobox = ctk.CTkComboBox(
            settings,
            variable=self.output_ratio_var,
            values=list(RATIO_OPTIONS),
            state="readonly",
        )
        self.output_ratio_combobox.grid(row=7, column=1, padx=(0, 14), pady=(0, 12), sticky="ew")

        controls = ctk.CTkFrame(self.content)
        controls.grid(row=3, column=0, padx=18, pady=8, sticky="ew")
        controls.grid_columnconfigure(2, weight=1)
        self.start_button = ctk.CTkButton(controls, text="开始批量处理", command=self._start, fg_color="#1f8f55", hover_color="#187443")
        self.start_button.grid(row=0, column=0, padx=(14, 8), pady=12)
        self.stop_button = ctk.CTkButton(controls, text="停止", command=self._stop, state="disabled", fg_color="#b43c3c", hover_color="#922f2f")
        self.stop_button.grid(row=0, column=1, padx=8, pady=12)
        self.status_label = ctk.CTkLabel(controls, text="等待开始")
        self.status_label.grid(row=0, column=2, padx=14, pady=12, sticky="e")

        status = ctk.CTkFrame(self.content)
        status.grid(row=4, column=0, padx=18, pady=8, sticky="ew")
        status.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(status, text="进度", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        self.progress = ctk.CTkProgressBar(status)
        self.progress.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="ew")
        self.progress.set(0)

        logs = ctk.CTkFrame(self.content)
        logs.grid(row=5, column=0, padx=18, pady=(8, 18), sticky="ew")
        logs.grid_columnconfigure(0, weight=1)
        logs.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(logs, text="运行日志", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        self.log_textbox = ctk.CTkTextbox(logs, wrap="word", height=220)
        self.log_textbox.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.log_textbox.tag_config("error", foreground="#ef5350")
        self.log_textbox.tag_config("success", foreground="#66bb6a")
        self.log_textbox.tag_config("warning", foreground="#ffb74d")

    def _refresh_image_list(self, select_all: bool = False) -> None:
        previous_selection = {path: variable.get() for path, variable in self.image_selection_vars.items()}
        for child in self.image_list_frame.winfo_children():
            child.destroy()
        self.image_selection_vars.clear()
        self.available_images = []

        input_path = self.input_path_var.get().strip()
        input_dir = Path(input_path) if input_path else Path()
        if input_path and input_dir.is_dir():
            try:
                self.available_images = sorted(
                    (item for item in input_dir.iterdir() if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS),
                    key=natural_sort_key,
                )
            except OSError as error:
                self._append_log("error", f"无法读取输入图片列表: {error}")

        for row, image_path in enumerate(self.available_images):
            selected = previous_selection.get(image_path, True if not previous_selection else select_all)
            variable = ctk.BooleanVar(value=selected)
            self.image_selection_vars[image_path] = variable
            checkbox = ctk.CTkCheckBox(
                self.image_list_frame,
                text=image_path.name,
                variable=variable,
                command=self._update_selection_count,
            )
            checkbox.grid(row=row, column=0, padx=8, pady=3, sticky="w")
        self._update_selection_count()

    def _update_selection_count(self) -> None:
        selected_count = sum(variable.get() for variable in self.image_selection_vars.values())
        self.selection_count_label.configure(text=f"已选择 {selected_count}/{len(self.available_images)} 张")

    def _select_all_images(self) -> None:
        for variable in self.image_selection_vars.values():
            variable.set(True)
        self._update_selection_count()

    def _clear_image_selection(self) -> None:
        for variable in self.image_selection_vars.values():
            variable.set(False)
        self._update_selection_count()

    def _selected_images(self) -> tuple[Path, ...]:
        return tuple(path for path in self.available_images if self.image_selection_vars.get(path) and self.image_selection_vars[path].get())

    def _choose_input(self) -> None:
        path = filedialog.askdirectory(title="选择输入文件夹")
        if path:
            self.input_path_var.set(path)
            self._refresh_image_list(select_all=True)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_path_var.set(path)

    def _update_denoise_label(self, value: float) -> None:
        self.denoise_label.configure(text=f"{value:.2f}")

    def _save_settings(self) -> None:
        data: dict[str, Any] = {
            "api_url": self.api_url_var.get().strip(),
            "remember_api_key": self.remember_api_key_var.get(),
            "model": self.model_var.get().strip(),
            "api_mode": self.api_mode_var.get(),
            "prompt": self.prompt_textbox.get("1.0", "end-1c"),
            "denoising_strength": round(self.denoise_var.get(), 2),
            "output_ratio": self.output_ratio_var.get(),
        }
        if self.remember_api_key_var.get():
            data["api_key"] = self.api_key_var.get()
        try:
            save_local_settings(data)
        except OSError as error:
            self._append_log("warning", f"无法保存本地设置: {error}")

    def _test_connection(self) -> None:
        if self.connection_worker and self.connection_worker.is_alive():
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在处理", "请在批量处理结束或停止后再测试连接。")
            return

        api_url = self.api_url_var.get().strip()
        api_key = self.api_key_var.get().strip()
        if not api_url.startswith(("http://", "https://")):
            messagebox.showerror("配置错误", "请先填写有效的 API URL。")
            return
        if not api_key:
            messagebox.showerror("配置错误", "请先填写 API Key。")
            return

        self._save_settings()
        self.test_connection_button.configure(state="disabled")
        self.status_label.configure(text="正在测试连接")
        self._append_log("info", "正在请求同一服务的 OpenAI 兼容模型列表...")
        self.connection_worker = ConnectionWorker(api_url, api_key, self.event_queue)
        self.connection_worker.start()

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()
        input_dir = Path(input_path) if input_path else Path()
        output_dir = Path(output_path) if output_path else Path()
        api_url = self.api_url_var.get().strip()
        api_key = self.api_key_var.get().strip()
        model = self.model_var.get().strip()
        api_mode = self.api_mode_var.get()
        output_ratio = self.output_ratio_var.get()
        prompt = self.prompt_textbox.get("1.0", "end-1c").strip()
        selected_files = self._selected_images()

        if not input_path or not input_dir.is_dir() or not output_path:
            messagebox.showerror("配置错误", "请选择有效的输入文件夹和输出文件夹。")
            return
        if not selected_files:
            messagebox.showerror("未选择图片", "请在“待处理图片”列表中至少勾选一张图片。")
            return
        if not api_url.startswith(("http://", "https://")):
            messagebox.showerror("配置错误", "API URL 必须以 http:// 或 https:// 开头。")
            return
        if not api_key:
            messagebox.showerror("配置错误", "请输入 API Key。")
            return
        if not model or model == "请先测试连接":
            messagebox.showerror("配置错误", "请先点击“测试连接并获取模型”，然后选择一个模型。")
            return
        if api_mode == OPENAI_IMAGE_EDIT_MODE and not urlsplit(api_url).path.rstrip("/").endswith("/images/edits"):
            messagebox.showerror(
                "接口路径错误",
                "OpenAI 图像编辑模式需要精确填写以 /images/edits 结尾的 URL，例如 https://xcode.best/v1/images/edits。",
            )
            return

        self._save_settings()
        config = JobConfig(
            input_dir,
            output_dir,
            selected_files,
            api_url,
            api_key,
            model,
            api_mode,
            prompt,
            round(self.denoise_var.get(), 2),
            output_ratio,
        )
        self.stop_event.clear()
        self.total_files = 0
        self.progress.set(0)
        self.log_textbox.delete("1.0", "end")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="正在处理")
        if api_mode == OPENAI_IMAGE_EDIT_MODE:
            self._append_log("info", "OpenAI 图像编辑模式将上传原图文件；该标准接口不使用重绘幅度参数。")
        self.worker = BatchWorker(config, self.stop_event, self.event_queue)
        self.worker.start()

    def _stop(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.status_label.configure(text="正在停止...")
            self._append_log("warning", "已收到停止请求，将在当前请求完成后停止。")

    def _append_log(self, level: str, message: str) -> None:
        self.log_textbox.insert("end", message + "\n", level if level != "info" else ())
        self.log_textbox.see("end")

    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self.event_queue.get_nowait()
                if event_type == "log":
                    level, message = payload
                    self._append_log(level, message)
                elif event_type == "set_total":
                    self.total_files = payload
                elif event_type == "progress" and self.total_files:
                    self.progress.set(payload / self.total_files)
                    self.status_label.configure(text=f"已处理 {payload}/{self.total_files}")
                elif event_type == "models_result":
                    success, message, models = payload
                    self.test_connection_button.configure(state="normal")
                    if success:
                        self.model_combobox.configure(values=models, state="readonly")
                        self.model_var.set(models[0])
                        self.status_label.configure(text="连接成功")
                        self._append_log("success", message)
                    else:
                        self.model_combobox.configure(values=["请先测试连接"], state="readonly")
                        self.model_var.set("请先测试连接")
                        self.status_label.configure(text="连接失败")
                        self._append_log("error", message)
                elif event_type == "finished":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_label.configure(text="已停止" if payload else "处理完成")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("仍在处理中", "正在处理图片。确定停止并关闭窗口吗？"):
                return
            self.stop_event.set()
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    ImageColoringApp().mainloop()
