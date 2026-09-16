"""Image file discovery, loading and small drawing helpers (numpy/PIL only)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(path: str, recursive: bool = True) -> List[str]:
    """Return sorted image files for a file path or a folder."""
    path = str(path or "").strip()
    if not path or not os.path.exists(path):
        return []
    if os.path.isfile(path):
        return [path] if os.path.splitext(path)[1].lower() in IMAGE_EXTS else []
    found = []
    for root, dirs, names in os.walk(path):
        dirs.sort()
        found.extend(os.path.join(root, n) for n in sorted(names) if os.path.splitext(n)[1].lower() in IMAGE_EXTS)
        if not recursive:
            break
    return found


def load_rgb(path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load an image as an HxWx3 uint8 RGB array resized to ``size=(W, H)``."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB").resize(size, Image.LANCZOS), dtype=np.uint8)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9\-_]+", "_", (text or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_") or "item"


def subject_name_from_path(path: str) -> str:
    """'golden_retriever_0a3f91.jpg' -> 'golden retriever' (drops ids and numbers)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w and not any(ch.isdigit() for ch in w)]
    return " ".join(words).lower() or "animal"


def format_prompt(template: str, animal: str) -> str:
    return (template or "").replace("{animals}", animal).replace("{animal}", animal)


def save_json(path: str, obj: Any) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def overlay_mask(background_rgb: np.ndarray, mask_hw: np.ndarray, color=(255, 64, 64), opacity: float = 0.6) -> np.ndarray:
    """Tint the background where the soft mask is active (visualization only)."""
    bg = background_rgb.astype(np.float32)
    weight = (np.clip(mask_hw, 0.0, 1.0) * opacity)[..., None]
    tint = np.array(color, dtype=np.float32)[None, None, :]
    return (bg * (1.0 - weight) + tint * weight).round().astype(np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def labeled_grid(
    items: Sequence[Tuple[Image.Image, str]],
    columns: int,
    tile: int = 256,
    caption_h: int = 28,
) -> Image.Image:
    """Tile images left-to-right with a caption under each one."""
    if not items:
        raise ValueError("labeled_grid needs at least one image")
    columns = max(1, min(int(columns), len(items)))
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile, rows * (tile + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(max(10, caption_h // 2))
    for idx, (img, caption) in enumerate(items):
        x, y = (idx % columns) * tile, (idx // columns) * (tile + caption_h)
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((tile, tile), Image.LANCZOS)
        canvas.paste(thumb, (x + (tile - thumb.width) // 2, y + (tile - thumb.height) // 2))
        text = caption if len(caption) <= 48 else caption[:45] + "..."
        draw.text((x + 6, y + tile + 6), text, fill="black", font=font)
    return canvas


def _wrap(text: str, width: int) -> List[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def comparison_sheet(
    cells: Dict[Tuple[int, int], str],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    tile: int = 256,
    max_rows: int = 24,
) -> Image.Image:
    """Rows = subject/background pairs, columns = variants; missing cells stay blank."""
    rows = min(len(row_labels), max_rows)
    label_w, header_h = 150, 44
    canvas = Image.new("RGB", (label_w + len(col_labels) * tile, header_h + rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for c, label in enumerate(col_labels):
        for k, text in enumerate(_wrap(label, 30)[:2]):
            draw.text((label_w + c * tile + 6, 4 + k * 18), text, fill="black", font=font)
    for r in range(rows):
        for k, text in enumerate(_wrap(row_labels[r], 18)[:6]):
            draw.text((6, header_h + r * tile + 8 + k * 18), text, fill="black", font=font)
        for c in range(len(col_labels)):
            path = cells.get((r, c))
            if path and os.path.exists(path):
                with Image.open(path) as img:
                    thumb = img.convert("RGB")
                    thumb.thumbnail((tile, tile), Image.LANCZOS)
                    canvas.paste(thumb, (label_w + c * tile, header_h + r * tile))
    return canvas


def thumbnail(path: str, size: int = 160) -> Image.Image:
    with Image.open(path) as img:
        thumb = img.convert("RGB")
        thumb.thumbnail((size, size), Image.LANCZOS)
        return thumb


def mask_to_rgba(mask_hw: np.ndarray, opacity: float = 0.82) -> Image.Image:
    """White mask with alpha = mask, plus a red outline; used by the layout canvas."""
    import cv2

    gray = (np.clip(mask_hw, 0.0, 1.0) * 255).astype(np.uint8)
    rgba = np.full((*gray.shape, 4), 255, dtype=np.uint8)
    rgba[..., 3] = (gray.astype(np.float32) * opacity).astype(np.uint8)
    outline = cv2.Canny(gray, 32, 128) > 0
    rgba[outline] = (255, 96, 96, 255)
    return Image.fromarray(rgba)


def describe_array(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 3:
        x = x.mean(axis=2)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "nonzero_ratio": float((x > 0).mean()),
    }


def first_image(path: str) -> Optional[str]:
    images = list_images(path)
    return images[0] if images else None
