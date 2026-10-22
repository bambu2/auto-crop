from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXIF_ORIENTATION_TAG = 0x0112
DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class CropResult:
    image: Image.Image
    margins: tuple[int, int, int, int] | None  # (top, bottom, left, right)

    @property
    def was_cropped(self) -> bool:
        return self.margins is not None


def _runs_at_least(mask: np.ndarray, min_run: int) -> bool:
    """True if mask contains a run of True of length >= min_run."""
    if min_run <= 1:
        return bool(mask.any())
    if mask.sum() < min_run:
        return False
    # Pad with False so diff sees boundaries; +1 on start, -1 on end.
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return bool(((ends - starts) >= min_run).any())


def crop_uniform_border(
    img: Image.Image,
    color_threshold: int = 20,
    buffer: int = 2,
    min_run: int = 2,
    max_scan_ratio: float = 0.5,
) -> CropResult:
    """Crop uniform borders while protecting edge-touching content.

    Returns a CropResult; margins is None if no crop was needed/applied.
    """
    if not (1 <= color_threshold <= 255):
        raise ValueError("color_threshold must be in [1, 255]")
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    if min_run < 1:
        raise ValueError("min_run must be >= 1")
    if not (0.0 < max_scan_ratio <= 1.0):
        raise ValueError("max_scan_ratio must be in (0, 1]")

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)
    h, w, _ = arr.shape
    if h < 3 or w < 3:
        return CropResult(img, None)

    # Background = median of corner patches.
    corner_size = max(3, min(h, w) // 50)
    corners = np.concatenate(
        [
            arr[:corner_size, :corner_size].reshape(-1, 3),
            arr[:corner_size, -corner_size:].reshape(-1, 3),
            arr[-corner_size:, :corner_size].reshape(-1, 3),
            arr[-corner_size:, -corner_size:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)

    def line_has_content(line: np.ndarray) -> bool:
        diff = np.abs(line - bg).max(axis=-1)
        return _runs_at_least(diff >= color_threshold, min_run)

    max_scan_h = max(1, int(h * max_scan_ratio))
    max_scan_w = max(1, int(w * max_scan_ratio))

    def scan(start: int, stop: int, step: int, getter) -> int:
        """Scan until content; return the index of last background line, or
        the far edge if none found within the scan window."""
        last_bg = start
        for i in range(start, stop, step):
            if line_has_content(getter(i)):
                return i
            last_bg = i
        return last_bg

    top_hit = scan(0, max_scan_h, 1, lambda i: arr[i])
    bottom_hit = scan(h - 1, h - 1 - max_scan_h, -1, lambda i: arr[i])
    left_hit = scan(0, max_scan_w, 1, lambda j: arr[:, j])
    right_hit = scan(w - 1, w - 1 - max_scan_w, -1, lambda j: arr[:, j])

    # Did we actually find content on each side?
    top_found = line_has_content(arr[top_hit])
    bottom_found = line_has_content(arr[bottom_hit])
    left_found = line_has_content(arr[:, left_hit])
    right_found = line_has_content(arr[:, right_hit])

    top = max(0, top_hit - buffer) if top_found else 0
    bottom = min(h - 1, bottom_hit + buffer) if bottom_found else h - 1
    left = max(0, left_hit - buffer) if left_found else 0
    right = min(w - 1, right_hit + buffer) if right_found else w - 1

    if top >= bottom or left >= right:
        return CropResult(img, None)
    if (top, bottom, left, right) == (0, h - 1, 0, w - 1):
        return CropResult(img, None)

    new_h, new_w = bottom - top + 1, right - left + 1
    if new_h < h * 0.1 or new_w < w * 0.1:
        return CropResult(img, None)
    if (h - new_h) + (w - new_w) < 4:
        return CropResult(img, None)

    cropped = img.crop((left, top, right + 1, bottom + 1))
    margins = (top, h - 1 - bottom, left, w - 1 - right)
    return CropResult(cropped, margins)


def _strip_orientation(exif: bytes | None) -> bytes | None:
    """Remove the orientation tag so EXIF-transposed pixels aren't rotated twice."""
    if not exif:
        return exif
    try:
        from PIL import Image as _Image

        exif_ifd = _Image.Exif()
        exif_ifd.load(exif)
        if EXIF_ORIENTATION_TAG in exif_ifd:
            del exif_ifd[EXIF_ORIENTATION_TAG]
        return exif_ifd.tobytes()
    except Exception:  # noqa: BLE001 - best-effort EXIF rewrite; fall back to original
        return exif


def safe_save(
    img: Image.Image,
    dst: Path,
    original_format: str | None,
    exif: bytes | None,
    quality: int | None = None,
) -> None:
    save_kwargs = {}
    if original_format:
        save_kwargs["format"] = original_format
    if exif:
        save_kwargs["exif"] = exif
    if original_format == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if quality is not None and original_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    img.save(dst, **save_kwargs)


_EXT_TO_FORMAT = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "jpe": "JPEG",
    "jfif": "JPEG",
    "png": "PNG",
    "bmp": "BMP",
    "tif": "TIFF",
    "tiff": "TIFF",
    "webp": "WEBP",
    "gif": "GIF",
}


def batch_crop(
    input_dir: str,
    output_dir: str,
    color_threshold: int = 20,
    buffer: int = 2,
    min_run: int = 2,
    max_scan_ratio: float = 0.5,
    exts: tuple[str, ...] = DEFAULT_EXTS,
    skip_existing: bool = True,
    recursive: bool = True,
    quality: int | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"input_dir not a directory: {input_path}")
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in exts)
    if not files:
        print(f"[!] No images found under {input_dir}")
        return {"total": 0, "cropped": 0, "no_crop": 0, "skipped": 0, "failed": 0}

    total = len(files)
    counts = {"cropped": 0, "no_crop": 0, "skipped": 0, "failed": 0}
    print(f"[*] {total} image(s), starting...\n" if verbose else "", end="")

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(input_path)
        dst = output_path / rel
        prefix = f"[{idx}/{total}]"

        if skip_existing and dst.exists():
            counts["skipped"] += 1
            if verbose:
                print(f"{prefix} skip (exists): {rel}")
            continue

        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img)
                original_format = img.format or _EXT_TO_FORMAT.get(
                    src.suffix.lstrip(".").lower()
                )
                exif = _strip_orientation(img.info.get("exif"))
                img_copy = img.copy()

            result = crop_uniform_border(
                img_copy, color_threshold, buffer, min_run, max_scan_ratio
            )

            if result.was_cropped:
                out_img = result.image
                counts["cropped"] += 1
            else:
                out_img = img_copy
                counts["no_crop"] += 1

            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                safe_save(out_img, dst, original_format, exif, quality)

            if verbose:
                if result.was_cropped:
                    t, b, l, r = result.margins
                    print(
                        f"{prefix} cropped -> {rel} "
                        f"(t{t} b{b} l{l} r{r}) "
                        f"{img_copy.size} -> {out_img.size}"
                    )
                else:
                    print(f"{prefix} no crop  -> {rel}")

        except Exception as e:  # noqa: BLE001 - per-file isolation is intentional
            counts["failed"] += 1
            print(f"{prefix} FAILED -> {rel}: {type(e).__name__}: {e}", file=sys.stderr)

    print("\n" + "=" * 50)
    print(
        f"Done. total={total} cropped={counts['cropped']} "
        f"no_crop={counts['no_crop']} skipped={counts['skipped']} "
        f"failed={counts['failed']}"
    )
    if not dry_run:
        print(f"Output: {output_path.resolve()}")
    return {"total": total, **counts}


def _cli() -> None:
    p = argparse.ArgumentParser(description="Batch-crop uniform image borders.")
    p.add_argument("input_dir", nargs="?", default="input")
    p.add_argument("output_dir", nargs="?", default="output")
    p.add_argument("--color-threshold", type=int, default=20)
    p.add_argument("--buffer", type=int, default=2)
    p.add_argument("--min-run", type=int, default=2)
    p.add_argument("--max-scan-ratio", type=float, default=0.5)
    p.add_argument(
        "--quality", type=int, default=None, help="JPEG/WEBP quality when re-encoding"
    )
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    batch_crop(
        a.input_dir,
        a.output_dir,
        color_threshold=a.color_threshold,
        buffer=a.buffer,
        min_run=a.min_run,
        max_scan_ratio=a.max_scan_ratio,
        skip_existing=not a.no_skip_existing,
        recursive=not a.no_recursive,
        quality=a.quality,
        dry_run=a.dry_run,
        verbose=not a.quiet,
    )


if __name__ == "__main__":
    _cli()
