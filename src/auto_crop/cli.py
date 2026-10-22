import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXIF_ORIENTATION_TAG = 0x0112
DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_QUALITY = 95  # re-encode quality for cropped JPEG/WEBP images


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


def _validate_params(
    color_threshold: int,
    buffer: int,
    min_run: int,
    max_scan_ratio: float,
) -> None:
    if not (1 <= color_threshold <= 255):
        raise ValueError("color_threshold must be in [1, 255]")
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    if min_run < 1:
        raise ValueError("min_run must be >= 1")
    if not (0.0 < max_scan_ratio <= 1.0):
        raise ValueError("max_scan_ratio must be in (0, 1]")


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
    _validate_params(color_threshold, buffer, min_run, max_scan_ratio)

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
        exif_ifd = Image.Exif()
        exif_ifd.load(exif)
        if EXIF_ORIENTATION_TAG not in exif_ifd:
            return exif
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
    if original_format == "JPEG" and img.mode not in ("RGB", "L", "CMYK"):
        img = img.convert("RGB")
    if original_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality if quality is not None else DEFAULT_QUALITY
    icc_profile = img.info.get("icc_profile")
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    dpi = img.info.get("dpi")
    if dpi and all(d > 0 for d in dpi):
        save_kwargs["dpi"] = dpi
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
    _validate_params(color_threshold, buffer, min_run, max_scan_ratio)
    if quality is not None and not (1 <= quality <= 100):
        raise ValueError("quality must be in [1, 100]")
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"input_dir not a directory: {input_path}")
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in exts)
    in_resolved = input_path.resolve()
    out_resolved = output_path.resolve()
    if out_resolved != in_resolved and out_resolved.is_relative_to(in_resolved):
        # Never re-ingest our own output when it lives inside the input tree.
        files = [p for p in files if not p.resolve().is_relative_to(out_resolved)]
    if not files:
        if verbose:
            print(f"[!] No images found under {input_dir}")
        return {"total": 0, "cropped": 0, "no_crop": 0, "skipped": 0, "failed": 0}

    total = len(files)
    counts = {"cropped": 0, "no_crop": 0, "skipped": 0, "failed": 0}
    if verbose:
        print(f"[*] {total} image(s), starting...\n")

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(input_path)
        dst = output_path / rel
        prefix = f"[{idx}/{total}]"

        if skip_existing and dst.is_file() and dst.stat().st_size > 0:
            counts["skipped"] += 1
            if verbose:
                print(f"{prefix} skip (exists): {rel}")
            continue

        try:
            with Image.open(src) as opened:
                img = ImageOps.exif_transpose(opened)
                original_format = (
                    _EXT_TO_FORMAT.get(src.suffix.lstrip(".").lower()) or opened.format
                )
                exif = _strip_orientation(img.info.get("exif"))

            result = crop_uniform_border(
                img, color_threshold, buffer, min_run, max_scan_ratio
            )
            reencode = result.was_cropped or quality is not None

            if result.was_cropped:
                out_img = result.image
                counts["cropped"] += 1
            else:
                out_img = img
                counts["no_crop"] += 1

            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if reencode:
                    safe_save(out_img, dst, original_format, exif, quality)
                elif src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)  # byte-identical, metadata preserved

            if verbose:
                margins = result.margins
                if margins is not None:
                    t, b, l, r = margins
                    print(
                        f"{prefix} cropped -> {rel} "
                        f"(t{t} b{b} l{l} r{r}) "
                        f"{img.size} -> {out_img.size}"
                    )
                else:
                    if reencode:
                        note = "re-encoded"
                    elif src.resolve() == dst.resolve():
                        note = "unchanged"
                    elif dry_run:
                        note = "would copy"
                    else:
                        note = "copied"
                    print(f"{prefix} no crop  -> {rel} ({note})")

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


def main() -> None:
    p = argparse.ArgumentParser(description="Batch-crop uniform image borders.")
    p.add_argument("input_dir", nargs="?", default="input")
    p.add_argument("output_dir", nargs="?", default="output")
    p.add_argument("--color-threshold", type=int, default=20)
    p.add_argument("--buffer", type=int, default=2)
    p.add_argument("--min-run", type=int, default=2)
    p.add_argument("--max-scan-ratio", type=float, default=0.5)
    p.add_argument(
        "--quality",
        type=int,
        default=None,
        help=(
            "Re-encode quality (1-100) for JPEG/WEBP. Default: copy uncropped "
            "files as-is and save cropped files at "
            f"{DEFAULT_QUALITY}; setting this forces re-encoding at N."
        ),
    )
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress; summary still prints",
    )
    a = p.parse_args()
    try:
        counts = batch_crop(
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
    except (ValueError, NotADirectoryError) as e:
        p.error(str(e))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
