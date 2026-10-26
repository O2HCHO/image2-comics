from __future__ import annotations

import base64
from contextlib import ExitStack
from dataclasses import dataclass
import mimetypes
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import requests
from .config import EDIT_MODE, JSON_MODE, ORIGINAL
from .imaging import png_bytes


@dataclass(frozen=True)
class ApiConfig:
    url: str
    key: str
    model: str
    prompt: str
    mode: str = EDIT_MODE
    denoise: float = 0.55
    ratio: str = ORIGINAL
    references: tuple[Path, ...] = ()
    timeout: float = 300
    preserve_text: bool = False

    def validate(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("请填写有效的 http(s) API 地址，不要在地址中嵌入密钥")
        if parsed.query or parsed.fragment:
            raise ValueError("API 地址不能包含查询参数或片段，请通过 API Key 字段提供密钥")
        if not self.key.strip() or not self.model.strip() or not self.prompt.strip():
            raise ValueError("请填写 API Key、模型和上色提示词")
        if self.mode not in {EDIT_MODE, JSON_MODE}:
            raise ValueError("不支持的接口模式")
        if self.mode == EDIT_MODE and not parsed.path.rstrip("/").endswith("/images/edits"):
            raise ValueError("图像编辑地址需以 /images/edits 结尾")
        if self.mode == JSON_MODE and self.references:
            raise ValueError("自定义 JSON 模式不支持参考图，请移除参考图后重试")


def safe_error(error: object, key: str = "") -> str:
    text = str(error)
    if key:
        text = text.replace(key, "[密钥已隐藏]")
    text = re.sub(r"https?://[^\s<>\"']+", "[服务地址]", text)
    text = re.sub(r"(?i)bearer\s+\S+|sk-[\w-]+", "[凭据已隐藏]", text)
    return text[:800]


def response_json(response: requests.Response) -> dict:
    if not response.ok:
        try:
            payload = response.json()
            detail = payload.get("error", payload.get("message", "服务请求失败"))
            if isinstance(detail, dict):
                detail = detail.get("message", "服务请求失败")
        except (ValueError, AttributeError):
            detail = "服务未返回 JSON 错误信息，请检查接口路径"
        raise ValueError(f"HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
    except ValueError as error:
        raise ValueError("接口未返回 JSON，可能填写了官网或文档地址") from error
    if not isinstance(data, dict):
        raise ValueError("API 返回的 JSON 不是对象")
    return data


def image_value(data: dict) -> tuple[str, str]:
    value = data.get("image")
    if not value and isinstance(data.get("images"), list) and data["images"]:
        value = data["images"][0]
    nested = data.get("data")
    if not value and isinstance(nested, list) and nested:
        value = nested[0]
    if not value and isinstance(nested, dict):
        value = nested.get("image") or nested.get("b64_json") or nested.get("url")
        if not value and isinstance(nested.get("images"), list) and nested["images"]:
            value = nested["images"][0]
    if not value:
        value = data.get("b64_json") or data.get("url")
    if isinstance(value, dict):
        if value.get("b64_json"):
            return "base64", value["b64_json"]
        value = value.get("url") or value.get("image")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("API 响应中没有图片数据")
    return ("url" if value.startswith(("http://", "https://")) else "base64"), value


class ImageApi:
    def color(self, source: Path, config: ApiConfig) -> bytes:
        config.validate()
        headers = {"Authorization": f"Bearer {config.key}"}
        try:
            if config.mode == EDIT_MODE:
                with ExitStack() as stack:
                    paths = (source,) + config.references
                    field = "image[]" if config.references else "image"
                    files = [(field, (path.name, stack.enter_context(path.open("rb")),
                                      mimetypes.guess_type(path.name)[0] or "application/octet-stream")) for path in paths]
                    prompt = config.prompt
                    if config.references:
                        prompt += "\n第一张图片是唯一需要上色的漫画页；后续图片仅作角色配色参考。只返回第一张漫画页的上色结果，禁止拼图或复制参考图内容。"
                    response = requests.post(config.url, headers=headers, files=files,
                                             data={"model": config.model, "prompt": prompt, "n": "1", "output_format": "png"},
                                             timeout=config.timeout, allow_redirects=False)
            else:
                response = requests.post(config.url, headers=headers,
                                         json={"model": config.model, "prompt": config.prompt,
                                               "image": base64.b64encode(source.read_bytes()).decode("ascii"),
                                               "denoising_strength": config.denoise},
                                         timeout=config.timeout, allow_redirects=False)
            with response:
                if response.is_redirect:
                    raise ValueError("API 返回重定向，请填写最终图像编辑接口地址")
                kind, value = image_value(response_json(response))
            if kind == "url":
                parsed = urlsplit(value)
                if not parsed.netloc or parsed.username or parsed.password:
                    raise ValueError("结果图片地址无效")
                with requests.get(value, timeout=config.timeout) as download:
                    if not download.ok:
                        raise ValueError(f"结果图下载失败：HTTP {download.status_code}")
                    content = download.content
            else:
                if value.startswith("data:"):
                    value = value.split(",", 1)[-1]
                try:
                    content = base64.b64decode(value, validate=True)
                except ValueError as error:
                    raise ValueError("返回的图片 Base64 无效") from error
            return png_bytes(content, ORIGINAL if config.preserve_text else config.ratio)
        except requests.Timeout as error:
            raise ValueError("请求超时；服务端可能仍在生成，请确认后手动重试以避免重复计费") from error
        except requests.RequestException as error:
            raise ValueError("网络请求失败：" + safe_error(error, config.key)) from error
        except ValueError as error:
            hint = "；若服务不支持多图，请移除参考图后重试" if config.references else ""
            raise ValueError(safe_error(error, config.key) + hint) from error

    def models(self, url: str, key: str) -> list[str]:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("请先填写有效的 API 地址")
        match = re.search(r"/v1(?:/|$)", parsed.path, re.I)
        prefix = parsed.path[:match.end()].rstrip("/") if match else "/v1"
        errors = []
        for path in dict.fromkeys((prefix + "/models", "/models")):
            endpoint = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
            try:
                with requests.get(endpoint, headers={"Authorization": f"Bearer {key}"}, timeout=20, allow_redirects=False) as response:
                    payload = response_json(response)
                models = payload.get("data") or payload.get("models") or []
                result = list(dict.fromkeys(item.get("id") if isinstance(item, dict) else item for item in models))
                result = [item for item in result if isinstance(item, str) and item.strip()]
                if result:
                    return result
                errors.append("模型列表为空")
            except (ValueError, requests.RequestException) as error:
                errors.append(safe_error(error, key))
        raise ValueError("；".join(errors) + "。仍可手动填写模型。")
