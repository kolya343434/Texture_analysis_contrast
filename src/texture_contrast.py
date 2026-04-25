from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GLCMConfig:
    levels: int = 16
    distance: int = 1
    angles_deg: tuple[int, ...] = (0, 45, 90, 135)
    symmetric: bool = True
    normalized: bool = True


def load_grayscale(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def quantize(image_u8: np.ndarray, levels: int) -> np.ndarray:
    if image_u8.dtype != np.uint8:
        raise TypeError("Expected uint8 grayscale image.")
    if levels <= 1 or levels > 256:
        raise ValueError("levels must be in [2..256].")
    # Map [0..255] -> [0..levels-1]
    return (image_u8.astype(np.uint16) * levels // 256).astype(np.uint16)


def _offset_from_angle(distance: int, angle_deg: int) -> tuple[int, int]:
    angle_rad = math.radians(angle_deg)
    dx = int(round(math.cos(angle_rad) * distance))
    dy = int(round(-math.sin(angle_rad) * distance))
    if dx == 0 and dy == 0:
        raise ValueError("Invalid (distance, angle) produced zero offset.")
    return dx, dy


def glcm(
    image_q: np.ndarray,
    *,
    levels: int,
    dx: int,
    dy: int,
    symmetric: bool = True,
    normalized: bool = True,
) -> np.ndarray:
    if image_q.ndim != 2:
        raise ValueError("Expected 2D grayscale image.")

    h, w = image_q.shape
    x0 = max(0, -dx)
    x1 = min(w, w - dx)
    y0 = max(0, -dy)
    y1 = min(h, h - dy)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Offset too large for the given image.")

    src = image_q[y0:y1, x0:x1]
    nbr = image_q[y0 + dy : y1 + dy, x0 + dx : x1 + dx]

    mat = np.zeros((levels, levels), dtype=np.float64)
    # Vectorized histogram via bincount on flattened pairs.
    idx = (src.astype(np.int64) * levels + nbr.astype(np.int64)).ravel()
    counts = np.bincount(idx, minlength=levels * levels).astype(np.float64)
    mat[:] = counts.reshape((levels, levels))

    if symmetric:
        mat = mat + mat.T

    if normalized:
        s = mat.sum()
        if s > 0:
            mat = mat / s

    return mat


def contrast_from_glcm(p: np.ndarray) -> float:
    levels = p.shape[0]
    i = np.arange(levels, dtype=np.float64)
    diff2 = (i[:, None] - i[None, :]) ** 2
    return float(np.sum(diff2 * p))


def contrast_features(image_u8: np.ndarray, cfg: GLCMConfig) -> dict[str, float]:
    image_q = quantize(image_u8, cfg.levels)
    results: dict[str, float] = {}
    for angle in cfg.angles_deg:
        dx, dy = _offset_from_angle(cfg.distance, angle)
        p = glcm(
            image_q,
            levels=cfg.levels,
            dx=dx,
            dy=dy,
            symmetric=cfg.symmetric,
            normalized=cfg.normalized,
        )
        results[f"contrast_{angle}deg"] = contrast_from_glcm(p)
    results["contrast_mean"] = float(np.mean(list(results.values()))) if results else 0.0
    return results


def generate_demo_textures(out_dir: str | Path, *, size: int = 256) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:size, 0:size]

    checker = (((x // 16) % 2) ^ ((y // 16) % 2)).astype(np.uint8) * 255
    stripes = (((x // 8) % 2)).astype(np.uint8) * 255
    noise = (rng.random((size, size)) * 255).astype(np.uint8)
    gradient = np.clip((x / (size - 1)) * 255, 0, 255).astype(np.uint8)

    images = {
        "checker.png": checker,
        "stripes.png": stripes,
        "noise.png": noise,
        "gradient.png": gradient,
    }

    written: list[Path] = []
    for name, arr in images.items():
        p = out_path / name
        Image.fromarray(arr, mode="L").save(p)
        written.append(p)
    return written
