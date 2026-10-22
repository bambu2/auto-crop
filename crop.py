from pathlib import Path

import numpy as np
from PIL import Image


def crop_uniform_border(
    img:Image.Image, color_threshold=20, buffer=2, min_run=2, max_scan_ratio=0.5
):
    """
    裁剪四周接近纯色的边缘，且防止裁掉贴边文字。

    参数:
        color_threshold: 像素与背景色的差异阈值（越小越敏感）
        buffer: 检测到内容后回退的像素数（避免切到抗锯齿边缘）
        min_run: 连续多少个非背景像素才算"有内容"
        max_scan_ratio: 最多扫描到图片尺寸的多少比例
    """
    img = img.convert("RGB")   # 统一转 RGB 再处理
    arr = np.array(img)
    h, w, _ = arr.shape

    # 用四角小块的中位数作为背景参考色（避开边缘文字）
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
    bg_color = np.median(corners, axis=0)

    def line_has_content(line):
        diff = np.abs(line.astype(int) - bg_color.astype(int)).max(axis=-1)
        mask = (diff >= color_threshold).astype(np.int8)
        if mask.sum() < min_run:
            return False
        # 向量化找连续 min_run 个 1
        kernel = np.ones(min_run, dtype=np.int8)
        conv = np.convolve(mask, kernel, mode="valid")
        return (conv >= min_run).any()

    top, bottom, left, right = 0, h - 1, 0, w - 1
    max_scan_h = int(h * max_scan_ratio)
    max_scan_w = int(w * max_scan_ratio)

    # 顶部
    for i in range(max_scan_h):
        if line_has_content(arr[i]):
            top = max(0, i - buffer)
            break
        top = i + 1

    # 底部
    for i in range(h - 1, h - 1 - max_scan_h, -1):
        if line_has_content(arr[i]):
            bottom = min(h - 1, i + buffer)
            break
        bottom = i - 1

    # 左侧
    for j in range(max_scan_w):
        if line_has_content(arr[:, j]):
            left = max(0, j - buffer)
            break
        left = j + 1

    # 右侧
    for j in range(w - 1, w - 1 - max_scan_w, -1):
        if line_has_content(arr[:, j]):
            right = min(w - 1, j + buffer)
            break
        right = j - 1

    new_h = bottom - top + 1
    new_w = right - left + 1

    if new_h < h * 0.1 or new_w < w * 0.1:
        return None, None
    if (top, bottom, left, right) == (0, h - 1, 0, w - 1):
        return None, None

    cropped = img.crop((left, top, right + 1, bottom + 1))
    return cropped, (top, h - 1 - bottom, left, w - 1 - right)


def safe_save(img, dst, original_format, exif):
    save_kwargs = {}
    if original_format:
        save_kwargs["format"] = original_format
    if exif:
        save_kwargs["exif"] = exif
    if original_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(dst, **save_kwargs)


def batch_crop(
    input_dir,
    output_dir,
    color_threshold=20,
    buffer=2,
    min_run=2,
    max_scan_ratio=0.5,
    exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
    skip_existing=True,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]

    if not files:
        print(f"[!] 在 {input_dir} 下未找到图片文件")
        return

    total = len(files)
    cropped_cnt = skipped_cnt = failed_cnt = 0

    print(f"[*] 共找到 {total} 张图片，开始处理...\n")

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(input_path)
        dst = output_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and dst.exists():
            print(f"[{idx}/{total}] 跳过 (已存在): {rel}")
            skipped_cnt += 1
            continue

        try:
            with Image.open(src) as img:
                original_format = img.format or src.suffix.lstrip(".").upper()
                exif = img.info.get("exif")
                img_copy = img.copy()

            cropped, margins = crop_uniform_border(
                img_copy, color_threshold, buffer, min_run, max_scan_ratio
            )

            if cropped is None:
                safe_save(img_copy, dst, original_format, exif)
                print(f"[{idx}/{total}] 无需裁剪 → {rel}")
                skipped_cnt += 1
            else:
                safe_save(cropped, dst, original_format, exif)
                t, b, l, r = margins
                print(
                    f"[{idx}/{total}] 已裁剪 → {rel}  "
                    f"(上{t} 下{b} 左{l} 右{r})  "
                    f"{img_copy.size} → {cropped.size}"
                )
                cropped_cnt += 1

        except Exception as e:
            print(f"[{idx}/{total}] 失败 → {rel}: {type(e).__name__}: {e}")
            failed_cnt += 1

    print("\n" + "=" * 50)
    print(
        f"完成！总计: {total} | 已裁剪: {cropped_cnt} | "
        f"无需裁剪: {skipped_cnt} | 失败: {failed_cnt}"
    )
    print(f"输出目录: {output_path.resolve()}")


if __name__ == "__main__":
    batch_crop(
        input_dir="input",
        output_dir="output",
        color_threshold=20,  # 越小越敏感；有浅色文字可降到 10~15
        buffer=2,  # 防止切到抗锯齿边缘
        min_run=2,  # 连续 2 个非背景像素才算有内容
        max_scan_ratio=0.5,  # 最多扫描一半
        skip_existing=True,  # 断点续跑
    )
