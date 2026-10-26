from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import re

from PIL import Image, ImageOps
from .config import RATIOS, ORIGINAL

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
