from PIL import Image
import numpy as np
from pathlib import Path


def crop_uniform_border(
    img,
    color_threshold=20,
    edge_ratio_threshold=0.999,  # 提高到 99.9%
    sample_rows=5,
    buffer=2,  # 停止后再回退的像素缓冲
    min_content_pixels=1,
):  # 一行/列中非背景像素的最少个数
    """
    裁剪四周接近纯色的边缘。
    关键：只要一行/列中存在 >= min_content_pixels 个明显非背景像素，即视为有内容，停止裁剪。
    """
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape

    # ---- 用 4 个角的小块中位数作为背景参考色（避开边缘文字） ----
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
        """该行/列是否存在明显非背景像素"""
        diff = np.abs(line.astype(int) - bg_color.astype(int)).max(axis=-1)
        content_mask = diff >= color_threshold
        if content_mask.sum() >= min_content_pixels:
            # 允许少量孤立的边缘抗锯齿噪点：要求连续或数量较多
            # 简单策略：非背景像素总数超过 min_content_pixels 即算有内容
            return True
        return False

    def is_pure_border(line):
        """该行/列是否可视为纯色边缘（无内容）"""
        return not line_has_content(line)

    top, bottom, left, right = 0, h - 1, 0, w - 1

    # 顶部向下扫描
    for i in range(h // 2):
        if is_pure_border(arr[i]):
            top = i + 1
        else:
            top = max(0, i - buffer)  # 回退缓冲，避免切到内容
            break

    # 底部向上扫描
    for i in range(h - 1, h // 2, -1):
        if is_pure_border(arr[i]):
            bottom = i - 1
        else:
            bottom = min(h - 1, i + buffer)
            break

    # 左侧向右扫描
    for j in range(w // 2):
        if is_pure_border(arr[:, j]):
            left = j + 1
        else:
            left = max(0, j - buffer)
            break

    # 右侧向左扫描
    for j in range(w - 1, w // 2, -1):
        if is_pure_border(arr[:, j]):
            right = j - 1
        else:
            right = min(w - 1, j + buffer)
            break

    new_h = bottom - top + 1
    new_w = right - left + 1

    if new_h < h * 0.1 or new_w < w * 0.1:
        return None, None
    if (top, bottom, left, right) == (0, h - 1, 0, w - 1):
        return None, None

    cropped = img.crop((left, top, right + 1, bottom + 1))
    return cropped, (top, h - 1 - bottom, left, w - 1 - right)


def safe_save(img, dst, original_format, exif):
    """安全保存：exif 为 None 时不传该参数，避免 len(None) 崩溃"""
    save_kwargs = {}
    if original_format:
        save_kwargs["format"] = original_format
    if exif:  # 只有非 None 且非空才传
        save_kwargs["exif"] = exif
    # 若是 JPEG 且带 alpha，需转 RGB 避免保存失败
    if original_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(dst, **save_kwargs)


def batch_crop(
    input_dir,
    output_dir,
    color_threshold=30,
    edge_ratio_threshold=0.95,
    sample_rows=5,
    exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
    skip_existing=False,
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
                exif = img.info.get("exif")  # 可能是 None
                img_copy = img.copy()  # 脱离 with 后再用更安全

            cropped, margins = crop_uniform_border(
                img_copy, color_threshold, edge_ratio_threshold, sample_rows
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
        color_threshold=30,
        edge_ratio_threshold=0.95,
        sample_rows=5,
        skip_existing=True,
    )
