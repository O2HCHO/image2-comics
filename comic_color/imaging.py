from __future__ import annotations

import hashlib
import math
import logging
from io import BytesIO
import os
from pathlib import Path
import re

from PIL import Image, ImageOps
from .config import RATIOS, ORIGINAL

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ASPECT_POLICY = "source-aspect-v1"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        # EXIF orientation is also applied when decoding results and previews.
        return ImageOps.exif_transpose(image).size


def exact_source_size(source_size, generated_size) -> tuple[int, int]:
    """Nearest pixel area on the exact source-ratio integer lattice.

    Rounding width and height separately is insufficient for exact equality.
    Keep at least the source resolution; retain larger generated sizes when
    possible, without adding borders or cropping panel content.
    """
    width, height = source_size
    gw, gh = generated_size
    divisor = math.gcd(width, height)
    rw, rh = width // divisor, height // divisor
    ideal = math.sqrt(gw * gh / (rw * rh))
    low = max(divisor, int(ideal))
    multiple = min((low, low + 1), key=lambda k: abs(rw * rh * k * k - gw * gh))
    return rw * multiple, rh * multiple


def source_ratio_png(content: bytes, source: Path, ratio: str = ORIGINAL) -> bytes:
    """Enforce the original aspect ratio before accepting a generated result."""
    if ratio != ORIGINAL:
        raise ValueError("漫画上色输出已锁定原稿长宽比，请使用“保持原始比例”")
    source_size = image_size(source)
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError("服务返回的内容不是可解码的图片") from error
    with image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        target = exact_source_size(source_size, image.size)
        if image.size != target:
            logging.getLogger(__name__).info("锁定原稿比例 %s: %s -> %s（原稿 %s）", source.name, image.size, target, source_size)
            image = image.resize(target, Image.Resampling.LANCZOS)
        if image.width * source_size[1] != image.height * source_size[0]:
            raise ValueError("输出长宽比与原稿不一致，结果未采用")
        stream = BytesIO()
        image.save(stream, "PNG")
        return stream.getvalue()


def natural_key(value: str) -> tuple:
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", value))


def scan_images(root: Path, output: Path | None, recursive: bool = True) -> list[Path]:
    root = root.resolve()
    excluded = output.resolve() if output else None
    if excluded == root:
        raise ValueError("输入目录和输出目录不能相同")
    result = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()
                          and (Path(current) / name).resolve() != excluded and name != ".comic-color"]
        result.extend(Path(current) / name for name in files
                      if Path(name).suffix.casefold() in EXTENSIONS and not (Path(current) / name).is_symlink())
        if not recursive:
            break
    return sorted(result, key=lambda path: natural_key(path.relative_to(root).as_posix()))


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_bytes(content: bytes, ratio: str = ORIGINAL) -> bytes:
    try:
        with Image.open(BytesIO(content)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError("服务返回的内容不是可解码的图片") from error
    selected = RATIOS.get(ratio)
    if selected:
        rw, rh = selected
        width, height = image.size
        multiple = max((width + rw - 1) // rw, (height + rh - 1) // rh)
        canvas = Image.new("RGBA", (multiple * rw, multiple * rh), (0, 0, 0, 0))
        canvas.alpha_composite(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
        image = canvas
    stream = BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        source.draft("RGB", size)
        image = ImageOps.exif_transpose(source)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image.convert("RGB")
