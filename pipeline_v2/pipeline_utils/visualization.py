"""Visualization utilities for segmentation results."""

import cv2
import numpy as np
from scipy.ndimage import find_objects
from skimage.segmentation import find_boundaries


def normalize_image(
    image: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> np.ndarray:
    """Normalize image to 0-1 range using percentiles."""
    vmin, vmax = np.percentile(image, [pmin, pmax])
    if vmax == vmin:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


# Preset colors (RGB, 0-1 range)
COLORS = {
    "blue": (0.4, 0.7, 0.95),  # BOP_Blue style - gentle sky blue
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.5, 0.0),
}


def colorize(
    image: np.ndarray,
    color: tuple | str = "blue",
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> np.ndarray:
    """Apply a colormap to grayscale image (black to color).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    color : tuple or str
        RGB tuple (0-1 range) or preset name from COLORS dict.
    pmin, pmax : float
        Percentiles for normalization.

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3), float32, range [0, 1].
    """
    if isinstance(color, str):
        color = COLORS[color]

    gray = normalize_image(image, pmin, pmax)
    return np.stack(
        [gray * color[0], gray * color[1], gray * color[2]], axis=-1
    )


def label_to_rgb(
    masks: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Convert label mask to RGB image with random colors.

    Parameters
    ----------
    masks : np.ndarray
        Integer label mask (0 = background).
    seed : int
        Random seed for reproducible colors.

    Returns
    -------
    np.ndarray
        RGB image with shape (H, W, 3), dtype float32, range [0, 1].
    """
    rng = np.random.default_rng(seed)
    n_labels = masks.max()

    # Generate random colors (background is black)
    colors = np.zeros((n_labels + 1, 3), dtype=np.float32)
    if n_labels > 0:
        colors[1:] = rng.random((n_labels, 3)).astype(np.float32)

    return colors[masks]


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.4,
    seed: int = 42,
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> np.ndarray:
    """Overlay colored masks on grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    masks : np.ndarray
        Integer label mask.
    alpha : float
        Mask transparency (0 = invisible, 1 = opaque).
    seed : int
        Random seed for colors.
    pmin, pmax : float
        Percentiles for image normalization.

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3), float32, range [0, 1].
    """
    # Normalize grayscale to RGB
    gray = normalize_image(image, pmin, pmax)
    base = np.stack([gray, gray, gray], axis=-1)

    if masks.max() == 0:
        return base

    # Create colored mask overlay
    mask_rgb = label_to_rgb(masks, seed=seed)
    foreground = masks > 0

    # Blend where masks are present
    result = base.copy()
    result[foreground] = (1 - alpha) * base[foreground] + alpha * mask_rgb[
        foreground
    ]

    return result


def overlay_boundaries(
    image: np.ndarray,
    masks: np.ndarray,
    color: tuple = (1.0, 1.0, 0.0),
    thickness: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> np.ndarray:
    """Overlay mask boundaries on grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    masks : np.ndarray
        Integer label mask.
    color : tuple
        RGB color for boundaries.
    thickness : int
        Boundary thickness in pixels.
    pmin, pmax : float
        Percentiles for image normalization.

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3), float32, range [0, 1].
    """
    gray = normalize_image(image, pmin, pmax)
    result = np.stack([gray, gray, gray], axis=-1)

    if masks.max() == 0:
        return result

    boundaries = find_boundaries(masks, mode="outer")

    if thickness > 1:
        from scipy.ndimage import binary_dilation

        boundaries = binary_dilation(boundaries, iterations=thickness - 1)

    result[boundaries] = color

    return result


def composite_segmentation(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    nucleoli_masks: np.ndarray | None = None,
    nuclei_boundary_color: tuple = (0.0, 1.0, 0.0),
    nucleoli_boundary_color: tuple = (1.0, 0.0, 1.0),
    image_color: tuple | str | None = "blue",
    pmin: float = 0.5,
    pmax: float = 99.5,
) -> np.ndarray:
    """Create composite RGB showing nuclei and nucleoli as outlines.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (typically DAPI).
    nuclei_masks : np.ndarray
        Integer label mask for nuclei.
    nucleoli_masks : np.ndarray, optional
        Integer label mask for nucleoli.
    nuclei_boundary_color : tuple
        RGB color for nuclear boundaries.
    nucleoli_boundary_color : tuple
        RGB color for nucleoli boundaries.
    image_color : tuple, str, or None
        Color for the background image. Use None for grayscale,
        a preset name (e.g., "blue"), or RGB tuple.
    pmin, pmax : float
        Percentiles for image normalization.

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3), float32, range [0, 1].
    """

    if image_color is None:
        gray = normalize_image(image, pmin, pmax)
        result = np.stack([gray, gray, gray], axis=-1)
    else:
        result = colorize(image, color=image_color, pmin=pmin, pmax=pmax)

    # Draw nuclear boundaries
    if nuclei_masks.max() > 0:
        boundaries = find_boundaries(nuclei_masks, mode="thick")
        result[boundaries] = nuclei_boundary_color

    # Draw nucleoli boundaries on top
    if nucleoli_masks is not None and nucleoli_masks.max() > 0:
        nucleoli_boundaries = find_boundaries(nucleoli_masks, mode="thick")
        result[nucleoli_boundaries] = nucleoli_boundary_color

    nucslices = find_objects(nuclei_masks)

    # cv2.putText requires an 8-bit image; `result` up to this point is
    # float32 in [0, 1], so convert before drawing the nucleus number labels.
    result = to_uint8(result)

    for i, nuc in enumerate(nucslices, start=1):
        if nuc is None:
            continue

        sy, sx = nuc

        labx = int((sx.stop - sx.start) / 2) + sx.start
        laby = int((sy.stop - sy.start) / 2) + sy.start

        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        cv2.putText(
            result,
            text,
            (labx, laby),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 4,
        )
        cv2.putText(
            result,
            text,
            (labx, laby),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
        )

    return result


def montage_rgb(
    images: list[np.ndarray], ncols: int = 3, gap: int = 2
) -> np.ndarray:
    """Arrange multiple RGB images into a montage grid.

    Parameters
    ----------
    images : list of np.ndarray
        RGB images with shape (H, W, 3).
    ncols : int
        Number of columns in grid.
    gap : int
        Gap between images in pixels.

    Returns
    -------
    np.ndarray
        Montage RGB image.
    """
    if not images:
        return np.zeros((1, 1, 3), dtype=np.float32)

    # Ensure all images have the same shape
    shapes = [img.shape[:2] for img in images]
    H = max(s[0] for s in shapes)
    W = max(s[1] for s in shapes)

    n = len(images)
    nrows = (n + ncols - 1) // ncols

    montage_h = nrows * H + (nrows - 1) * gap
    montage_w = ncols * W + (ncols - 1) * gap
    montage = np.zeros((montage_h, montage_w, 3), dtype=np.float32)

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols
        y = row * (H + gap)
        x = col * (W + gap)
        h, w = img.shape[:2]
        montage[y : y + h, x : x + w] = img

    return montage


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float32 RGB image to uint8 for saving."""
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


# ============================================================================
# Matplotlib convenience wrappers (optional, for notebook use)
# ============================================================================


def show_rgb(
    image: np.ndarray,
    ax=None,
    title: str | None = None,
    figsize: tuple = (8, 8),
):
    """Display RGB image using matplotlib.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    return ax


def show_segmentation_panel(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    nucleoli_masks: np.ndarray,
    figsize: tuple = (14, 5),
):
    """Display three-panel view: original, nuclei, composite.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    nuclei_masks : np.ndarray
        Nuclei labels.
    nucleoli_masks : np.ndarray
        Nucleoli labels.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original
    gray = normalize_image(image)
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Nuclei boundaries
    nuclei_rgb = overlay_boundaries(image, nuclei_masks, color=(0, 1, 0))
    axes[1].imshow(nuclei_rgb)
    axes[1].set_title(f"Nuclei (n={nuclei_masks.max()})")
    axes[1].axis("off")

    # Composite
    composite = composite_segmentation(image, nuclei_masks, nucleoli_masks)
    axes[2].imshow(composite)
    axes[2].set_title(f"Nucleoli (n={nucleoli_masks.max()})")
    axes[2].axis("off")

    fig.tight_layout()
    return fig
