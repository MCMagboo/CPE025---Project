"""Image preprocessing for the AI models (OpenCV / Pillow). No UI or model code."""

import cv2
import numpy as np
from PIL import Image

# ImageNet mean as 0-255 RGB; effdet pads letterboxed images with this colour.
IMAGENET_MEAN_RGB = (124, 116, 104)


def to_rgb(image):
    """Both models take 3-channel RGB; grayscale X-rays are copied into each channel."""
    return image if image.mode == "RGB" else image.convert("RGB")


def letterbox(image, size, fill=IMAGENET_MEAN_RGB):
    """Shrink to fit a size x size square, keeping aspect, padded at bottom/right.

    Returns (square_image, scale); divide model boxes by `scale` to get
    original-image pixels.
    """
    scale = size / max(image.size)
    resized = image.resize(
        (max(round(image.width * scale), 1), max(round(image.height * scale), 1)),
        Image.Resampling.BILINEAR,
    )
    square = Image.new("RGB", (size, size), fill)
    square.paste(resized, (0, 0))
    return square, scale


def clahe(image, clip_limit=2.0, tile_grid=8):
    """CLAHE contrast enhancement followed by min-max normalization, as RGB."""
    gray = np.asarray(image.convert("L"))
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(gray)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(normalized).convert("RGB")
