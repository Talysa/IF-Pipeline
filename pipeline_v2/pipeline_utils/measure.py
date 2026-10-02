"""Measurements: nuclear intensity/CTCF, foci, compartments, and overlap.

All size settings are in microns and converted with the image's pixel size.
The foci and compartment methods are the same ones used in the previous
run_analysis_foci / run_analysis_nucleolus_foci scripts, consolidated here.
"""

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, white_tophat

from .core import um_to_px, um2_to_px


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _nucleus_crops(nuclei_mask: np.ndarray, pad: int):
    """Yield (nucleus_id, crop slices, boolean nucleus mask in the crop)."""
    H, W = nuclei_mask.shape
    for nuc_id, sl in enumerate(ndi.find_objects(nuclei_mask), start=1):
        if sl is None:
            continue
        sy, sx = sl
        crop = (slice(max(0, sy.start - pad), min(H, sy.stop + pad)),
                slice(max(0, sx.start - pad), min(W, sx.stop + pad)))
        yield nuc_id, crop, nuclei_mask[crop] == nuc_id


def _remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Keep connected objects (4-connected) with at least min_px pixels."""
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lbl]


def _fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Fill enclosed background holes smaller than max_hole_px.

    Background touching the crop edge is outside the nucleus, never a hole.
    """
    holes, n = ndi.label(~mask)
    if n == 0:
        return mask
    sizes = np.bincount(holes.ravel())
    edge_ids = np.unique(np.concatenate(
        [holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    fill = sizes < max_hole_px
    fill[0] = False
    fill[edge_ids] = False
    return mask | fill[holes]


def _robust_threshold(values: np.ndarray, k: float):
    """median + k * robust SD (from the MAD); None if no usable spread."""
    med = np.median(values)
    sd = 1.4826 * np.median(np.abs(values - med))
    if not np.isfinite(sd) or sd == 0:
        sd = np.std(values)
    if not np.isfinite(sd) or sd == 0:
        return None
    return med + k * sd


# -----------------------------------------------------------------------------
# Background and per-nucleus intensity
# -----------------------------------------------------------------------------
def estimate_backgrounds(nuclei_mask, channel_images, pixel_size_um,
                         dilation_um=1.3, percentile=10):
    """Per-channel background: a low percentile of pixels outside a dilated
    nuclear mask (avoids nuclei and the rim right around them)."""
    nuclear = nuclei_mask > 0
    dil_px = um_to_px(dilation_um, pixel_size_um, min_px=0)
    buffered = ndi.binary_dilation(nuclear, iterations=dil_px) if dil_px > 0 else nuclear
    bg_region = ~buffered
    if np.count_nonzero(bg_region) < 100:
        bg_region = ~nuclear

    out = {}
    for name, img in channel_images.items():
        vals = img[bg_region]
        vals = vals[np.isfinite(vals)]
        out[name] = float(np.percentile(vals, percentile)) if vals.size else np.nan
    return out


def measure_nuclei(nuclei_mask, channel_images, backgrounds, pixel_size_um):
    """One row per nucleus: shape, plus mean, total, background and CTCF for
    every channel. CTCF = total intensity - area_px * background."""
    if nuclei_mask.max() == 0:
        return pd.DataFrame()

    props = pd.DataFrame(regionprops_table(
        nuclei_mask,
        properties=("label", "area", "perimeter", "eccentricity",
                    "solidity", "equivalent_diameter_area", "centroid"),
    ))
    df = pd.DataFrame({
        "NUC_label": props["label"].astype(int),
        "nucleus_area_px": props["area"].astype(int),
        "nucleus_area_um2": props["area"] * pixel_size_um**2,
        "perimeter_um": props["perimeter"] * pixel_size_um,
        "equivalent_diameter_um": props["equivalent_diameter_area"] * pixel_size_um,
        "eccentricity": props["eccentricity"],
        "solidity": props["solidity"],
        "centroid_y": props["centroid-0"],
        "centroid_x": props["centroid-1"],
    })

    ids = df["NUC_label"].to_numpy()
    area = df["nucleus_area_px"].to_numpy()
    for name, img in channel_images.items():
        total = ndi.sum_labels(img, nuclei_mask, index=ids)
        bg = backgrounds.get(name, np.nan)
        df[f"{name}_mean_intensity"] = total / area
        df[f"{name}_min_intensity"] = ndi.minimum(img, nuclei_mask, index=ids)
        df[f"{name}_max_intensity"] = ndi.maximum(img, nuclei_mask, index=ids)
        df[f"{name}_total_intensity"] = total
        df[f"{name}_background"] = bg
        df[f"{name}_total_ctcf"] = total - area * bg if np.isfinite(bg) else np.nan

    return df


# -----------------------------------------------------------------------------
# Foci
# -----------------------------------------------------------------------------
def segment_foci(image, nuclei_mask, pixel_size_um, tophat_radius_um=0.9,
                 sigma_um=0.07, threshold_k=4.0, min_area_um2=0.15):
    """Detect foci inside each nucleus.

    Smooth, remove local background with a white top-hat, then threshold each
    nucleus at median + threshold_k * robust SD of its own top-hat signal.

    Returns (labelled foci mask, DataFrame with one row per focus).
    """
    sigma_px = sigma_um / pixel_size_um
    radius_px = um_to_px(tophat_radius_um, pixel_size_um)
    min_px = um2_to_px(min_area_um2, pixel_size_um)

    smoothed = ndi.gaussian_filter(image.astype(np.float32), sigma=sigma_px)
    enhanced = white_tophat(smoothed, footprint=disk(radius_px))

    foci_mask = np.zeros(nuclei_mask.shape, dtype=np.int32)
    rows = []
    next_id = 1

    for nuc_id, crop, nuc in _nucleus_crops(nuclei_mask, pad=2):
        vals = enhanced[crop][nuc]
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
            continue
        thr = _robust_threshold(vals, threshold_k)
        if thr is None:
            continue

        cand = (enhanced[crop] >= thr) & nuc
        cand = _remove_small(cand, min_px)
        lbl = label(cand)  # 8-connected, as before

        raw = image[crop]
        for local_id in range(1, lbl.max() + 1):
            obj = lbl == local_id
            if not obj.any():
                continue
            foci_mask[crop][obj] = next_id
            px = raw[obj]
            rows.append({
                "focus_id": next_id,
                "NUC_label": nuc_id,
                "focus_area_px": int(obj.sum()),
                "focus_area_um2": obj.sum() * pixel_size_um**2,
                "focus_total_intensity": float(px.sum()),
                "focus_mean_intensity": float(px.mean()),
                "focus_max_intensity": float(px.max()),
            })
            next_id += 1

    return foci_mask, pd.DataFrame(rows)


def overlap_with(foci_mask, other_mask, min_fraction=0.0):
    """For each focus: how many of its pixels fall inside other_mask, that
    as a fraction of the focus, and whether the fraction is above
    min_fraction (0 = any overlap counts).

    Returns a DataFrame indexed by focus_id.
    """
    ids = np.arange(1, foci_mask.max() + 1)
    if ids.size == 0:
        return pd.DataFrame(columns=["overlap_px", "overlap_fraction", "overlaps"])
    area = ndi.sum_labels(np.ones_like(foci_mask), foci_mask, index=ids)
    inside = ndi.sum_labels(other_mask > 0, foci_mask, index=ids)
    frac = np.divide(inside, area, out=np.zeros_like(inside, dtype=float), where=area > 0)
    return pd.DataFrame({"overlap_px": inside.astype(int), "overlap_fraction": frac,
                         "overlaps": frac > min_fraction},
                        index=pd.Index(ids, name="focus_id"))


# -----------------------------------------------------------------------------
# Compartments (nucleoli, speckles, ...)
# -----------------------------------------------------------------------------
def segment_compartments(image, nuclei_mask, pixel_size_um, sigma_um=0.13,
                         min_area_um2=0.75, hole_area_um2=0.25,
                         open_radius_um=0.13, close_radius_um=0.26):
    """Segment bright subnuclear regions of a marker (e.g. nucleoli from NCL,
    speckles from SC35).

    Smooth, Otsu-threshold within each nucleus, clean up with opening/closing,
    fill small holes, drop small objects.

    Returns (labelled mask, DataFrame with compartment_id -> NUC_label).
    """
    sigma_px = sigma_um / pixel_size_um
    open_px = um_to_px(open_radius_um, pixel_size_um, min_px=0)
    close_px = um_to_px(close_radius_um, pixel_size_um, min_px=0)
    min_px = um2_to_px(min_area_um2, pixel_size_um)
    hole_px = um2_to_px(hole_area_um2, pixel_size_um)

    smoothed = ndi.gaussian_filter(image.astype(np.float32), sigma=sigma_px)

    comp_mask = np.zeros(nuclei_mask.shape, dtype=np.int32)
    rows = []
    next_id = 1
    pad = max(open_px, close_px) + 3

    for nuc_id, crop, nuc in _nucleus_crops(nuclei_mask, pad=pad):
        if nuc.sum() < min_px:
            continue
        vals = smoothed[crop][nuc]
        vals = vals[np.isfinite(vals)]
        if vals.size < 10 or np.isclose(vals.min(), vals.max()):
            continue

        cand = (smoothed[crop] >= threshold_otsu(vals)) & nuc
        if open_px > 0:
            cand = ndi.binary_opening(cand, structure=disk(open_px))
        if close_px > 0:
            cand = ndi.binary_closing(cand, structure=disk(close_px))
        cand = _fill_small_holes(cand, hole_px)
        cand = _remove_small(cand, min_px)
        lbl = label(cand)

        for local_id in range(1, lbl.max() + 1):
            obj = lbl == local_id
            if not obj.any():
                continue
            comp_mask[crop][obj] = next_id
            rows.append({"compartment_id": next_id, "NUC_label": nuc_id})
            next_id += 1

    return comp_mask, pd.DataFrame(rows, columns=["compartment_id", "NUC_label"])


def measure_compartments(comp_mask, parents, channel_images, pixel_size_um):
    """One row per compartment: size, shape, and mean/min/max/total
    intensity per channel."""
    if comp_mask.max() == 0:
        return pd.DataFrame()
    props = pd.DataFrame(regionprops_table(
        comp_mask,
        properties=("label", "area", "perimeter", "eccentricity",
                    "solidity", "equivalent_diameter_area"),
    )).rename(columns={"label": "compartment_id"})
    props = parents.merge(props, on="compartment_id", how="left")
    df = pd.DataFrame({
        "compartment_id": props["compartment_id"],
        "NUC_label": props["NUC_label"],
        "compartment_area_px": props["area"].astype(int),
        "compartment_area_um2": props["area"] * pixel_size_um**2,
        "perimeter_um": props["perimeter"] * pixel_size_um,
        "equivalent_diameter_um": props["equivalent_diameter_area"] * pixel_size_um,
        "eccentricity": props["eccentricity"],
        "solidity": props["solidity"],
    })

    ids = df["compartment_id"].to_numpy()
    area = df["compartment_area_px"].to_numpy()
    for name, img in channel_images.items():
        total = ndi.sum_labels(img, comp_mask, index=ids)
        df[f"{name}_mean_intensity"] = total / area
        df[f"{name}_min_intensity"] = ndi.minimum(img, comp_mask, index=ids)
        df[f"{name}_max_intensity"] = ndi.maximum(img, comp_mask, index=ids)
        df[f"{name}_total_intensity"] = total
    return df


def compartment_signal_per_nucleus(nuclei_mask, comp_mask, channel_images,
                                   pixel_size_um, comp_name):
    """Per nucleus: compartment area and, for each channel, signal inside vs.
    outside the compartment. Column names use comp_name (e.g. 'nucleoli')."""
    ids = np.arange(1, nuclei_mask.max() + 1)
    inside_any = comp_mask > 0
    nuc_area = ndi.sum_labels(np.ones_like(nuclei_mask), nuclei_mask, index=ids)
    comp_area = ndi.sum_labels(inside_any, nuclei_mask, index=ids)

    df = pd.DataFrame({
        "NUC_label": ids,
        f"{comp_name}_area_px": comp_area.astype(int),
        f"{comp_name}_area_um2": comp_area * pixel_size_um**2,
        f"frac_{comp_name}_area": np.divide(comp_area, nuc_area,
                                            out=np.full(ids.size, np.nan), where=nuc_area > 0),
    })
    for name, img in channel_images.items():
        total = ndi.sum_labels(img, nuclei_mask, index=ids)
        inside = ndi.sum_labels(np.where(inside_any, img, 0), nuclei_mask, index=ids)
        frac_in = np.divide(inside, total, out=np.full(ids.size, np.nan), where=total > 0)
        df[f"{name}_in_{comp_name}_intensity"] = inside
        df[f"{name}_outside_{comp_name}_intensity"] = total - inside
        df[f"{name}_frac_in_{comp_name}"] = frac_in
        df[f"{name}_frac_outside_{comp_name}"] = 1 - frac_in
    return df
