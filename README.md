# auto-crop

Batch-crop uniform borders (solid backgrounds, margins, letterboxing) off images
while protecting content that touches the edge of the frame.

## Features

- Detects the border color from the image corners and trims the uniform border
  away.
- Only trims a side when the border there is finite and found within the scan
  window, so edge-to-edge content is left alone.
- Keeps untouched files pristine: images that need no crop are copied
  byte-for-byte instead of re-encoded.
- Preserves EXIF, ICC profile and DPI on images that are re-encoded; the EXIF
  orientation tag is stripped after pixels are transposed, so output is not
  rotated twice.
- Recursive scans, nested output directories, `--dry-run`, and per-file error
  isolation.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended — the project uses it for the
  virtualenv and locked dependencies)
- `numpy` and `Pillow` (installed by `uv sync`)

## Setup

```bash
uv sync
```

## Usage

```bash
# crop ./input into ./output with defaults
uv run auto-crop

# explicit directories
uv run auto-crop photos cropped

# preview without writing anything
uv run auto-crop photos cropped --dry-run

# re-encode everything (cropped or not) at quality 85
uv run auto-crop photos cropped --quality 85
```

`input_dir` and `output_dir` default to `input` and `output`.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `input_dir` | `input` | Directory scanned for images |
| `output_dir` | `output` | Destination directory; the input tree is mirrored |
| `--color-threshold` | `20` | Max per-channel difference (0-255) from the background color that still counts as background |
| `--buffer` | `2` | Pixels of background kept around the detected content on each side |
| `--min-run` | `2` | Minimum length of a run of non-background pixels for a line to be treated as content |
| `--max-scan-ratio` | `0.5` | Fraction of each dimension that may be scanned for content before giving up |
| `--quality` | unset | Re-encode quality (1-100) for JPEG/WEBP. Unset: copy uncropped files as-is, save cropped files at 95. Set: re-encode every file at N |
| `--no-skip-existing` | off | Overwrite existing outputs (default skips non-empty destinations) |
| `--no-recursive` | off | Scan only the top level of `input_dir` |
| `--dry-run` | off | Print what would happen without writing files |
| `--quiet` | off | Suppress per-file progress lines (the final summary still prints) |

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`.

### How cropping works

1. The background color is the median of small corner patches.
2. Each of the four sides is scanned inward, stopping at the first line that
   contains content (a run of pixels farther than `--color-threshold` from the
   background, at least `--min-run` long).
3. A side is trimmed only if content was actually found within the scan window;
   otherwise the original edge is kept.
4. A crop is rejected when it would remove fewer than 4 pixels in total, shrink
   either dimension below 10%, or produce an inverted crop box.

### Output behavior

- **No crop needed:** the input file is copied byte-for-byte (pixels, EXIF, ICC
  profile, DPI and mtime preserved), unless `--quality` is given, which forces a
  re-encode.
- **Cropped:** re-encoded in the source format; JPEG/WEBP use `--quality` or 95,
  and ICC profile, DPI and EXIF (minus the orientation tag) are preserved.
- Existing non-empty destination files are skipped by default; a zero-byte file
  left by an interrupted run is reprocessed.
- If `output_dir` is inside `input_dir`, files previously written under
  `output_dir` are excluded from the scan.
- Destination paths mirror the input tree; subdirectories are created as needed.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success, including "no images found" |
| `1` | At least one file failed to process |
| `2` | Invalid arguments (bad option value or missing input directory) |

## Library usage

```python
from PIL import Image

from crop import batch_crop, crop_uniform_border

with Image.open("photo.jpg") as img:
    result = crop_uniform_border(img, color_threshold=20)
if result.was_cropped:
    result.image.save("cropped.jpg", quality=95)

counts = batch_crop("photos", "out", dry_run=True)
# {'total': 3, 'cropped': 1, 'no_crop': 2, 'skipped': 0, 'failed': 0}
```

`crop_uniform_border` returns a `CropResult` with `.image` (the cropped or
original image) and `.margins`, a `(top, bottom, left, right)` tuple or `None`
when nothing was cropped.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
```

## Known limitations

- No automated test suite yet; `--dry-run` is the quickest sanity check.
