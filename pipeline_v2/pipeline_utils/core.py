"""Loading images, calibration, and nucleus segmentation.

One pipeline handles both resolutions: the pixel size is passed in as a
setting instead of living in two separate copies of this file.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.segmentation import relabel_sequential

# Mac OpenMP workaround (harmless elsewhere)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
# Microns per pixel for each export resolution, plus the image size each one
# should have. The size check catches the old silent failure of analysing
# 1024px images with 512px calibration (or the reverse).
PIXEL_SIZE_UM = {"512px": 0.2631, "1024px": 0.1315306}
EXPECTED_SHAPE = {"512px": (512, 512), "1024px": (1024, 1024)}


def um_to_px(length_um: float, pixel_size_um: float, min_px: int = 1) -> int:
    """Convert a length (e.g. a radius) in microns to whole pixels."""
    return max(min_px, int(round(length_um / pixel_size_um)))


def um2_to_px(area_um2: float, pixel_size_um: float) -> int:
    """Convert an area in square microns to pixels (rounded up)."""
    return max(1, int(np.ceil(area_um2 / pixel_size_um**2)))


# -----------------------------------------------------------------------------
# Cellpose model
# -----------------------------------------------------------------------------
_MODEL = None
_MODEL_NAME = None
DEFAULT_MODEL = "cpsam_v2"


def load_model(pretrained_model: str = DEFAULT_MODEL, gpu: bool = True):
    """Load a Cellpose model once and reuse it.

    pretrained_model can be a built-in name ("cpsam_v2", "cpdino", ...) or the
    full path to a custom model such as cpsam_dapi_v1.
    """
    global _MODEL, _MODEL_NAME
    from cellpose.models import CellposeModel  # imported here so the rest of
    # the package can be imported (and tested) without Cellpose installed

    _MODEL = CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    _MODEL_NAME = pretrained_model
    return _MODEL


def current_model_name():
    return _MODEL_NAME


# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------
def load_image(path, expected_shape=None) -> np.ndarray:
    """Load a Fiji-exported TIF and return a (C, Y, X) float32 array.

    Accepts (C, Y, X) or (1, C, Y, X). Anything else raises an error rather
    than guessing which axis is which.
    """
    img = tifffile.imread(path)

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3:
        raise ValueError(
            f"{Path(path).name}: unexpected TIF shape {img.shape}; "
            "expected (channels, Y, X)."
        )

    if expected_shape is not None and tuple(img.shape[-2:]) != tuple(expected_shape):
        raise ValueError(
            f"{Path(path).name} is {img.shape[-2]}x{img.shape[-1]} px but the "
            f"selected resolution expects {expected_shape[0]}x{expected_shape[1]}. "
            "Check that the resolution setting matches these images."
        )

    return img.astype(np.float32)


# File names: dash-separated fields, described by a "name format" such as
#   date-antibody_1-antibody_2-treatment-timepoint-rep-image_number
# Each field becomes a column in the CSVs. "date" means YYYY-MM-DD.
# Fiji projection prefixes like "MAX_" are ignored.
DEFAULT_NAME_FORMAT = "date-antibody_1-antibody_2-treatment-timepoint-rep-image_number"
PROJECTION_PREFIXES = ("MAX_", "AVG_", "SUM_", "MIN_", "STD_", "MED_")
_RESERVED = {"filename", "NUC_label", "nucleolus_id", "focus_id", "marker"}


def name_fields(name_format: str) -> list:
    """Split a name format into field names, checking they make valid columns."""
    fields = [f.strip() for f in name_format.split("-")]
    for f in fields:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", f):
            raise ValueError(
                f"File name format field {f!r}: use letters, numbers and _ only, "
                "starting with a letter (fields are separated by -)")
        if f in _RESERVED:
            raise ValueError(f"{f!r} can't be used as a file name field")
    if len(set(fields)) != len(fields):
        raise ValueError(f"File name format repeats a field: {name_format}")
    return fields


@lru_cache(maxsize=None)
def _name_regex(name_format: str):
    parts = [rf"(?P<{f}>\d{{4}}-\d{{2}}-\d{{2}})" if f == "date" else rf"(?P<{f}>[^-]+)"
             for f in name_fields(name_format)]
    return re.compile("^" + "-".join(parts) + "$")


def parse_filename(stem: str, name_format: str = DEFAULT_NAME_FORMAT) -> dict:
    """Split a file name into condition columns using name_format.

    Names that don't match get empty values (the filename column is always
    kept, so nothing is lost).
    """
    for prefix in PROJECTION_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    m = _name_regex(name_format).match(stem)
    if m is None:
        return {f: None for f in name_fields(name_format)}
    return m.groupdict()


def natural_sort(paths):
    """Sort so image2 comes before image10."""
    def key(p):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", Path(p).name)]
    return sorted(paths, key=key)


# -----------------------------------------------------------------------------
# Nucleus segmentation
# -----------------------------------------------------------------------------
def remove_edge_nuclei(masks: np.ndarray, border_px: int = 5):
    """Remove nuclei whose bounding box comes within border_px of the edge.

    Returns (relabelled masks, number of nuclei removed).
    """
    if border_px <= 0:
        return masks, 0

    H, W = masks.shape
    out = masks.copy()
    removed = 0
    for label_id, sl in enumerate(ndi.find_objects(out), start=1):
        if sl is None:
            continue
        sy, sx = sl
        if (sy.start < border_px or sx.start < border_px
                or sy.stop > H - border_px or sx.stop > W - border_px):
            region = out[sl]
            region[region == label_id] = 0
            removed += 1

    out, _, _ = relabel_sequential(out)
    return out, removed


def segment_nuclei(
    dapi: np.ndarray,
    pixel_size_um: float,
    diameter_um: float | None = 25.0,
    remove_border_px: int = 5,
):
    """Segment nuclei from a 2D DAPI image with the loaded Cellpose model.

    diameter_um=None lets Cellpose run on the image at its original size.
    Returns (masks, number of nuclei removed at the border).
    """
    if _MODEL is None:
        load_model()

    diameter_px = None if diameter_um is None else diameter_um / pixel_size_um
    masks, *_ = _MODEL.eval(dapi, diameter=diameter_px)
    masks = np.asarray(masks, dtype=np.int32)

    return remove_edge_nuclei(masks, remove_border_px)
