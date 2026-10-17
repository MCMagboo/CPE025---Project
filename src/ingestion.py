"""X-ray file ingestion for OrthoScan AI: PNG, JPEG and DICOM.

No UI code here. load_xray() turns a file into an 8-bit grayscale ("L") or
RGB PIL image ready to display, or raises XrayLoadError with a message the
clinician can act on. Only pixels are kept: DICOM header details (patient
name, ID, dates) are discarded, and nothing is copied or saved.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DICOM_EXTENSIONS = {".dcm", ".dicom"}
FILE_TYPES = [
    ("X-ray images", "*.png *.jpg *.jpeg *.dcm *.dicom"),
    ("All files", "*.*"),
]


class XrayLoadError(Exception):
    """A file couldn't be loaded; the message is shown to the clinician."""


def load_xray(path):
    """Load an X-ray image file. Blocking; run it off the UI thread."""
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            raise _error(path, "the file is empty.")
        if is_dicom(path):
            return _load_dicom(path)
        return _load_picture(path)
    except FileNotFoundError:
        raise _error(path, "the file no longer exists.") from None
    except PermissionError:
        raise _error(
            path, "Windows denied access to it. Check that it isn't open in another program."
        ) from None
    except IsADirectoryError:
        raise _error(path, "it's a folder, not an image file.") from None


def is_dicom(path):
    """DICOM by extension, or by the "DICM" marker scanners write at byte 128."""
    path = Path(path)
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    with path.open("rb") as f:
        f.seek(128)
        return f.read(4) == b"DICM"


def _error(path, reason):
    return XrayLoadError(f"Couldn't open {Path(path).name}: {reason}")


# --- PNG / JPEG (anything Pillow reads) ---------------------------------------
def _load_picture(path):
    try:
        with Image.open(path) as img:
            img.load()
            # Phone photos often store rotation as a tag instead of rotating pixels.
            img = ImageOps.exif_transpose(img)
    except UnidentifiedImageError:
        raise _error(path, "it isn't a PNG, JPEG or DICOM image.") from None
    except Image.DecompressionBombError:
        raise _error(path, "the image is too large to open safely.") from None
    except (OSError, SyntaxError, ValueError):
        raise _error(path, "the file is damaged or incomplete.") from None

    if img.mode in ("L", "RGB"):
        return img
    if img.mode == "1":
        return img.convert("L")
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # 16-bit X-rays: stretch the used range to 8 bits for display.
        return Image.fromarray(_to_uint8(np.asarray(img)))
    if img.mode == "LA":
        return img.convert("L")
    return img.convert("RGB")  # palette, alpha, CMYK, YCbCr…


# --- DICOM -----------------------------------------------------------------------
def _load_dicom(path):
    import pydicom
    from pydicom.errors import InvalidDicomError
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    forced = False
    try:
        try:
            ds = pydicom.dcmread(path)
        except InvalidDicomError:
            # Some exports drop the 128-byte preamble; only trust that for .dcm files.
            if path.suffix.lower() not in DICOM_EXTENSIONS:
                raise
            ds = pydicom.dcmread(path, force=True)
            forced = True
    except InvalidDicomError:
        raise _error(path, "it isn't a PNG, JPEG or DICOM image.") from None
    except Exception:  # noqa: BLE001 - pydicom raises many types for broken files
        raise _error(path, "the DICOM file is damaged or incomplete.") from None

    if "PixelData" not in ds:
        if forced:
            # A forced read parses almost anything; without an image it wasn't DICOM.
            raise _error(path, "it isn't a PNG, JPEG or DICOM image.")
        raise _error(
            path, "it's a DICOM file without an image in it (for example a report)."
        )

    try:
        pixels = ds.pixel_array
    except Exception:  # noqa: BLE001
        syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if syntax is not None and syntax.is_compressed:
            raise _error(
                path, f"it uses a compression format OrthoScan can't read yet ({syntax.name})."
            ) from None
        raise _error(path, "the DICOM image couldn't be decoded; the file may be damaged.") from None

    frames = int(ds.get("NumberOfFrames", 1) or 1)
    samples = int(ds.get("SamplesPerPixel", 1) or 1)
    if frames > 1:
        pixels = pixels[0]

    if samples == 3:
        image = Image.fromarray(_to_uint8(pixels))
    else:
        # Use the scanner's own rescale and window/level so it looks as intended.
        pixels = apply_modality_lut(pixels, ds)
        pixels = apply_voi_lut(pixels, ds, index=0)
        gray = _to_uint8(pixels)
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            gray = 255 - gray  # MONOCHROME1 stores bone as dark; flip so bone is white
        image = Image.fromarray(gray)

    del ds  # drop the header (patient details) as soon as the pixels are out
    return image


def _to_uint8(array):
    """Min-max scale any numeric array to 0-255."""
    array = np.asarray(array, dtype=np.float32)
    low, high = float(np.nanmin(array)), float(np.nanmax(array))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - low) * (255.0 / (high - low))
    return np.nan_to_num(scaled).clip(0, 255).astype(np.uint8)
