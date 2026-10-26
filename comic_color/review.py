"""Resolve the one displayed result per input page, including legacy outputs."""
from pathlib import Path
from .imaging import EXTENSIONS, fingerprint, valid_image


def result_path(root: Path, output: Path, source: Path, record: dict) -> Path | None:
    output = output.resolve()
    relative = record.get("display_output")
    if not relative and record.get("status") == "success":
        relative = record.get("output")
    if relative:
        path = (output / relative).resolve()
        if path.is_relative_to(output) and path.is_file():
            return path
        # A tracked missing result must not silently turn into a legacy version.
        return None
    rel = source.resolve().relative_to(root.resolve())
    siblings = [p for p in source.parent.iterdir() if p.stem.casefold() == source.stem.casefold()
                and p.suffix.casefold() in EXTENSIONS]
    names = {f"colored_{rel.name}"} if len(siblings) == 1 or source.suffix.lower() != ".png" else set()
    if len(siblings) == 1:
        names.add(f"colored_{rel.stem}.png")
    candidates = [(output / rel.parent / name).resolve() for name in names]
    candidates = [path for path in candidates if path.is_relative_to(output) and path.is_file() and path != source.resolve()]
    return candidates[0] if len(candidates) == 1 else None


def export_pages(root: Path, output: Path, files: tuple[Path, ...], records: dict) -> tuple[list[Path], list[str]]:
    """Keep caller's natural input order; never export historical duplicates."""
    pages, missing = [], []
    for source in files:
        relative = source.resolve().relative_to(root.resolve()).as_posix()
        record = records.get(relative, {})
        path = result_path(root, output, source, record)
        if path is None:
            missing.append(relative)
            continue
        if not valid_image(path):
            raise ValueError(f"上色结果损坏，导出已停止：{relative}")
        expected = record.get("display_hash") or record.get("output_hash")
        if expected and fingerprint(path) != expected:
            raise ValueError(f"上色结果被外部修改，请核对后重新生成：{relative}")
        source_hash = record.get("display_source_hash") or record.get("source_hash")
        if source_hash and fingerprint(source) != source_hash:
            raise ValueError(f"原稿已改变，当前上色结果过期：{relative}")
        pages.append(path)
    return pages, missing
