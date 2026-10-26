"""One current colored image per page, at native aspect ratio, losslessly."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable

from PIL import Image, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def export_pdf(pages: list[Path], destination: Path, progress: Callable | None = None) -> int:
    if not pages:
        raise ValueError("没有可导出的上色结果")
    destination = destination.resolve()
    if destination.suffix.lower() != ".pdf":
        raise ValueError("导出文件扩展名必须为 .pdf")
    if any(path.resolve() == destination for path in pages):
        raise ValueError("PDF 路径不能覆盖图片")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix=".comic-pdf-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            document = canvas.Canvas(stream, pageCompression=1)
            document.setTitle(destination.stem)
            document.setCreator("Image2 Comics")
            for index, path in enumerate(pages):
                with Image.open(path) as source:
                    source.load()
                    image = ImageOps.exif_transpose(source).convert("RGBA")
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    # 144 dpi controls physical page size, never resamples pixels.
                    width, height = background.width / 2, background.height / 2
                    document.setPageSize((width, height))
                    document.drawImage(ImageReader(background), 0, 0, width=width, height=height)
                    document.showPage()
                if progress:
                    progress(index + 1, len(pages))
            document.save()
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return len(pages)
