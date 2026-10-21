from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def trim_conservative(image_path, tol=12.0, run=8, pad=12):
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]

    def edge_color(s):
        return s.reshape(-1, 3).mean(axis=0)

    def is_bg(line, ref):
        return np.abs(line - ref).mean() <= tol

    def scan(lines, ref):
        count = 0
        for line in lines:
            l = line.mean(axis=0) if line.ndim > 1 else line
            if is_bg(l, ref):
                count += 1
            else:
                break
        return max(0, count - pad)

    top_ref = edge_color(arr[:2])
    bottom_ref = edge_color(arr[-2:])
    left_ref = edge_color(arr[:, :2])
    right_ref = edge_color(arr[:, -2:])

    top = scan(arr[: h // 3], top_ref)
    bottom = h - scan(arr[::-1][: h // 3], bottom_ref)
    left = scan(arr[:, : w // 3].transpose(1, 0, 2), left_ref)
    right = w - scan(arr[:, ::-1][:, : w // 3].transpose(1, 0, 2), right_ref)

    top, left = max(0, top), max(0, left)
    bottom, right = min(h, bottom), min(w, right)
    if bottom - top < 10 or right - left < 10:
        return img
    return img.crop((left, top, right, bottom))


def _worker(args):
    src, dst, tol, pad = args
    try:
        r = trim_conservative(src, tol=tol, pad=pad)
        dst.parent.mkdir(parents=True, exist_ok=True)
        r.save(dst)
        return f"OK  {src.name} -> {r.size}"
    except Exception as e:
        return f"FAIL {src.name}: {e}"


def batch(input_dir, output_dir, tol, pad, jobs):
    in_p, out_p = Path(input_dir), Path(output_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    tasks = [
        (f, out_p / f.relative_to(in_p), tol, pad)
        for f in in_p.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]
    print(f"共 {len(tasks)} 张")
    if jobs <= 1:
        for t in tasks:
            print(_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for fut in as_completed([ex.submit(_worker, t) for t in tasks]):
                print(fut.result())


def preview(src, tol, pad):
    out = trim_conservative(src, tol=tol, pad=pad)
    dst = Path(f"preview_{Path(src).stem}_tol{tol}_pad{pad}.png")
    out.save(dst)
    print(f"{src} tol={tol} pad={pad} -> {out.size}  {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", metavar="IMG")
    p.add_argument("--tol", type=float, default=12.0)
    p.add_argument("--pad", type=int, default=12)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("input", nargs="?")
    p.add_argument("output", nargs="?")
    a = p.parse_args()

    if a.preview:
        for tol in (8, 12, 20):
            for pad in (8, 12, 20):
                preview(a.preview, tol, pad)
    elif a.input and a.output:
        batch(a.input, a.output, a.tol, a.pad, a.jobs)
    else:
        p.print_help()


def _gradient_profile(side: np.ndarray) -> np.ndarray:
    """side: (N, H, W) 由外向内的切片序列，返回每个切片的平均梯度"""
    flat = side.reshape(side.shape[0], -1)
    return np.abs(np.diff(flat, axis=1)).mean(axis=1)


def _scan_smooth(profile: np.ndarray, grad_thresh: float, run: int, pad: int) -> int:
    """从外向内，连续 run 个切片梯度 < thresh 才认定为背景，遇到高梯度立即停"""
    count = 0
    for g in profile:
        if g < grad_thresh:
            count += 1
        else:
            break
    return max(0, count - pad)


def trim_gradient(
    image_path: Path,
    grad_thresh: float = 1.5,  # 梯度阈值，越小越激进
    run: int = 4,  # 连续平滑切片数
    pad: int = 3,  # 回退像素
) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # 每边最多扫 1/4，防止极端情况全裁
    n = max(4, min(h, w) // 4)

    top = _scan_smooth(_gradient_profile(arr[:n]), grad_thresh, run, pad)
    bottom = _scan_smooth(_gradient_profile(arr[::-1][:n]), grad_thresh, run, pad)
    left = _scan_smooth(
        _gradient_profile(arr.transpose(1, 0, 2)[:n]), grad_thresh, run, pad
    )
    right = _scan_smooth(
        _gradient_profile(arr[:, ::-1].transpose(1, 0, 2)[:n]), grad_thresh, run, pad
    )

    bottom, right = h - bottom, w - right
    if bottom - top < 10 or right - left < 10:
        return img
    return img.crop((left, top, right, bottom))
