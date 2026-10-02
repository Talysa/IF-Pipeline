"""Run the analysis on one image or a whole folder.

Outputs (in OUTPUT_DIR):
    csvs/<experiment>_images.csv    one row per image (nuclei counted, removed at border)
    csvs/<experiment>_nuclei.csv    one row per nucleus (intensity, CTCF, foci, compartment)
    csvs/<experiment>_foci.csv      one row per focus (if foci channels are set; 'marker' column)
    csvs/<experiment>_<name>.csv    one row per compartment, e.g. _nucleoli.csv or
                                    _speckles.csv (if a compartment channel is set)
    check/   QC overlays       masks/   label masks       run_log.json
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imwrite
try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **kwargs):
        return it

from . import core, measure
from .visualization import composite_segmentation

TABLES = ("images", "nuclei", "foci", "compartment")  # internal names


@dataclass
class Settings:
    """Everything that controls an analysis run.

    channels: marker name -> channel index for every non-DAPI channel to
    measure. foci_channels / compartment_channel / overlap_pair refer to those
    names. Channel indices start at 0.

    A "compartment" is a bright subnuclear region outlined by a marker, e.g.
    nucleoli (NCL) or speckles (SC35). compartment_name is used in the output
    file and column names (n_nucleoli, BP53_in_nucleoli_intensity, ...).
    """
    resolution: str = "512px"
    name_format: str = core.DEFAULT_NAME_FORMAT  # how file names split into columns
    dapi_channel: int = 2
    channels: dict = field(default_factory=lambda: {"NCL": 1, "BP53": 0})

    foci_channels: list = field(default_factory=lambda: ["BP53"])
    compartment_channel: str | None = "NCL"
    compartment_name: str = "nucleoli"
    overlap_pair: tuple | None = None  # e.g. ("gH2AX", "BP53")

    # nuclei
    nucleus_diameter_um: float | None = 25.0
    remove_border_px: int = 5

    # background for CTCF
    background_dilation_um: float = 1.3
    background_percentile: float = 10

    # foci (defaults for every foci channel)
    foci_tophat_radius_um: float = 0.9
    foci_sigma_um: float = 0.07
    foci_threshold_k: float = 4.0
    foci_min_area_um2: float = 0.15
    # per-channel changes, e.g. {"BP53": {"threshold_k": 5.0}}
    foci_overrides: dict = field(default_factory=dict)

    # a focus counts as overlapping (another marker's foci, or a compartment)
    # when more than this fraction of its pixels overlap; 0 = any pixel
    overlap_min_fraction: float = 0.0

    # compartment segmentation (defaults tuned for nucleoli)
    compartment_sigma_um: float = 0.13
    compartment_min_area_um2: float = 0.75
    compartment_hole_area_um2: float = 0.25
    compartment_open_radius_um: float = 0.13
    compartment_close_radius_um: float = 0.26

    def __post_init__(self):
        if self.resolution not in core.PIXEL_SIZE_UM:
            raise ValueError(f"resolution must be one of {list(core.PIXEL_SIZE_UM)}")
        core.name_fields(self.name_format)  # raises if the format is invalid
        if isinstance(self.overlap_pair, list):
            self.overlap_pair = tuple(self.overlap_pair)

        indices = [self.dapi_channel, *self.channels.values()]
        if len(set(indices)) != len(indices):
            raise ValueError(f"Two channels share an index: DAPI={self.dapi_channel}, "
                             f"markers={self.channels}")
        if "dapi" in self.channels:
            raise ValueError("Don't list DAPI in channels; set dapi_channel instead.")
        for name in self.foci_channels:
            if name not in self.channels:
                raise ValueError(f"Foci channel {name!r} is not in channels {list(self.channels)}")
        if self.compartment_channel is not None:
            if self.compartment_channel not in self.channels:
                raise ValueError(f"Compartment marker {self.compartment_channel!r} is not in "
                                 f"channels {list(self.channels)}")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.compartment_name or ""):
                raise ValueError(f"Compartment name {self.compartment_name!r}: use letters, "
                                 "numbers and _ only, starting with a letter (e.g. nucleoli)")
            if self.compartment_name in (*TABLES, "NUC"):
                raise ValueError(f"{self.compartment_name!r} can't be used as a compartment name")
        if self.overlap_pair is not None:
            if len(self.overlap_pair) != 2 or not set(self.overlap_pair) <= set(self.foci_channels):
                raise ValueError("overlap_pair needs two names that are both foci channels")
        for name in self.foci_overrides:
            if name not in self.foci_channels:
                raise ValueError(f"foci_overrides has {name!r}, which is not a foci channel")

    @property
    def pixel_size_um(self):
        return core.PIXEL_SIZE_UM[self.resolution]

    def foci_params(self, name):
        p = {
            "tophat_radius_um": self.foci_tophat_radius_um,
            "sigma_um": self.foci_sigma_um,
            "threshold_k": self.foci_threshold_k,
            "min_area_um2": self.foci_min_area_um2,
        }
        p.update(self.foci_overrides.get(name, {}))
        return p


# -----------------------------------------------------------------------------
# One image
# -----------------------------------------------------------------------------
def analyze_image(img: np.ndarray, s: Settings):
    """Analyse one (C, Y, X) image. Returns (tables, masks, check images)."""
    px = s.pixel_size_um
    n_ch = img.shape[0]
    for idx in [s.dapi_channel, *s.channels.values()]:
        if idx >= n_ch:
            raise ValueError(f"Channel {idx} requested but the image has {n_ch} channels")

    dapi = img[s.dapi_channel]
    markers = {name: img[idx] for name, idx in s.channels.items()}
    all_channels = {"dapi": dapi, **markers}

    # --- nuclei ---
    nuclei, n_removed = core.segment_nuclei(
        dapi, px, diameter_um=s.nucleus_diameter_um, remove_border_px=s.remove_border_px)
    backgrounds = measure.estimate_backgrounds(
        nuclei, all_channels, px, s.background_dilation_um, s.background_percentile)
    nuc_df = measure.measure_nuclei(nuclei, all_channels, backgrounds, px)

    images_df = pd.DataFrame([{"n_nuclei": int(nuclei.max()),
                               "n_nuclei_removed_at_border": n_removed}])
    masks = {"NUC": nuclei}

    # --- compartment (nucleoli, speckles, ...) ---
    comp = None
    comp_df = pd.DataFrame()
    cname = s.compartment_name
    if s.compartment_channel is not None:
        comp, parents = measure.segment_compartments(
            markers[s.compartment_channel], nuclei, px,
            sigma_um=s.compartment_sigma_um, min_area_um2=s.compartment_min_area_um2,
            hole_area_um2=s.compartment_hole_area_um2,
            open_radius_um=s.compartment_open_radius_um,
            close_radius_um=s.compartment_close_radius_um)
        masks[f"{cname}_from_{s.compartment_channel}"] = comp
        comp_df = measure.measure_compartments(comp, parents, all_channels, px)

        if len(nuc_df):
            counts = parents.groupby("NUC_label").size() if len(parents) else pd.Series(dtype=int)
            nuc_df[f"n_{cname}"] = nuc_df["NUC_label"].map(counts).fillna(0).astype(int)
            per_nuc = measure.compartment_signal_per_nucleus(nuclei, comp, all_channels, px, cname)
            nuc_df = nuc_df.merge(per_nuc, on="NUC_label", how="left")

    # --- foci ---
    foci_masks, foci_tables = {}, []
    for name in s.foci_channels:
        fmask, fdf = measure.segment_foci(markers[name], nuclei, px, **s.foci_params(name))
        foci_masks[name] = fmask
        masks[f"{name}_foci"] = fmask
        if len(fdf):
            fdf.insert(0, "marker", name)
            if comp is not None:
                ov = measure.overlap_with(fmask, comp, s.overlap_min_fraction)
                fdf[f"focus_overlap_{cname}_px"] = fdf["focus_id"].map(ov["overlap_px"])
                fdf[f"focus_overlap_{cname}_fraction"] = fdf["focus_id"].map(ov["overlap_fraction"])
                fdf[f"in_{cname}"] = fdf["focus_id"].map(ov["overlaps"])
        foci_tables.append((name, fdf))

    if s.overlap_pair is not None:
        a, b = s.overlap_pair
        for this, other in ((a, b), (b, a)):
            fdf = dict(foci_tables)[this]
            if len(fdf):
                ov = measure.overlap_with(foci_masks[this], foci_masks[other],
                                          s.overlap_min_fraction)
                # generic column names so both markers share columns in foci.csv
                fdf["overlap_partner"] = other
                fdf["partner_overlap_px"] = fdf["focus_id"].map(ov["overlap_px"])
                fdf["partner_overlap_fraction"] = fdf["focus_id"].map(ov["overlap_fraction"])
                fdf["overlaps_partner_focus"] = fdf["focus_id"].map(ov["overlaps"])

    # per-nucleus foci counts
    if len(nuc_df):
        for name, fdf in foci_tables:
            grp = fdf.groupby("NUC_label") if len(fdf) else None

            def count(col=None, grp=grp):
                if grp is None:
                    return 0
                c = grp.size() if col is None else grp[col].sum()
                return nuc_df["NUC_label"].map(c).fillna(0).astype(int)

            nuc_df[f"n_{name}_foci"] = count()
            if comp is not None:
                n_in = count(f"in_{cname}")
                nuc_df[f"n_{name}_foci_in_{cname}"] = n_in
                nuc_df[f"n_{name}_foci_outside_{cname}"] = nuc_df[f"n_{name}_foci"] - n_in
                nuc_df[f"frac_{name}_foci_in_{cname}"] = (
                    n_in / nuc_df[f"n_{name}_foci"].replace(0, np.nan))
            if s.overlap_pair is not None and name in s.overlap_pair:
                other = [x for x in s.overlap_pair if x != name][0]
                n_ov = count("overlaps_partner_focus")
                nuc_df[f"n_{name}_foci_overlapping_{other}"] = n_ov
                nuc_df[f"frac_{name}_foci_overlapping_{other}"] = (
                    n_ov / nuc_df[f"n_{name}_foci"].replace(0, np.nan))

    # foci touching each compartment
    if len(comp_df):
        for name in s.foci_channels:
            fm = foci_masks[name]
            both = (comp > 0) & (fm > 0)
            pairs = pd.DataFrame({"compartment_id": comp[both], "focus_id": fm[both]})
            per = pairs.drop_duplicates().groupby("compartment_id").size()
            comp_df[f"n_{name}_foci"] = comp_df["compartment_id"].map(per).fillna(0).astype(int)

    foci_df = pd.concat([f for _, f in foci_tables if len(f)], ignore_index=True) \
        if any(len(f) for _, f in foci_tables) else pd.DataFrame()

    # --- QC overlays ---
    qc = dict(pmin=0.1, pmax=99.9, nuclei_boundary_color=(1, 0, 0))
    checks = {"DAPI": composite_segmentation(dapi, nuclei, comp, image_color="blue",
                                             nucleoli_boundary_color=(0.85, 0.63, 0.19), **qc)}
    if s.compartment_channel is not None:
        checks[s.compartment_channel] = composite_segmentation(
            markers[s.compartment_channel], nuclei, comp, image_color="green",
            nucleoli_boundary_color=(0.85, 0.63, 0.19), **qc)
    for name in s.foci_channels:
        checks[f"{name}_foci"] = composite_segmentation(
            markers[name], nuclei, foci_masks[name], image_color="magenta",
            nucleoli_boundary_color=(0.2, 1.0, 1.0), **qc)

    tables = {"images": images_df, "nuclei": nuc_df, "foci": foci_df, "compartment": comp_df}
    return tables, masks, checks


# -----------------------------------------------------------------------------
# A folder
# -----------------------------------------------------------------------------
def _versions():
    import importlib.metadata as md
    out = {}
    for pkg in ("cellpose", "numpy", "scipy", "scikit-image", "pandas", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _experiment_name(image_dir: Path) -> str:
    # images usually sit in <experiment>/TIF, so use the experiment folder name
    return image_dir.parent.name if image_dir.name.upper().startswith("TIF") else image_dir.name


def run_folder(image_dir, settings: Settings, output_dir=None, experiment_name=None,
               skip_existing=True, save_checks=True, save_masks=True):
    """Analyse every TIF in image_dir.

    Results for each image are saved as they finish, so an interrupted run
    picks up where it stopped (skip_existing=True). Combined CSVs are written
    at the end.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    experiment_name = experiment_name or _experiment_name(image_dir)

    csv_dir, check_dir, mask_dir = output_dir / "csvs", output_dir / "check", output_dir / "masks"
    per_image_dir = csv_dir / "per_image"
    for d in (per_image_dir, check_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = core.natural_sort(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".tif", ".tiff") and not p.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No TIF files found in {image_dir}")

    if core.current_model_name() is None:
        core.load_model()

    # --- run log: refuse to mix settings within one output folder ---
    log_path = output_dir / "run_log.json"
    this_run = {"settings": asdict(settings), "model": core.current_model_name()}
    this_run["settings"]["overlap_pair"] = (list(settings.overlap_pair)
                                            if settings.overlap_pair else None)
    if log_path.exists() and skip_existing:
        previous = json.loads(log_path.read_text())
        old = {"model": previous.get("model"), **previous.get("settings", {})}
        new = {"model": this_run["model"], **this_run["settings"]}
        show = lambda d, k: repr(d[k]) if k in d else "(not set)"
        changes = [f"  {k}: {show(old, k)} -> {show(new, k)}"
                   for k in sorted(set(old) | set(new), key=lambda k: k != "model")
                   if old.get(k, "(not set)") != new.get(k, "(not set)")]
        if changes:
            raise RuntimeError(
                f"{output_dir} already has results made with different settings:\n"
                + "\n".join(changes)
                + "\nUse a new results folder (in the notebook: set RUN_LABEL in Step 2) "
                "or delete the old one, then run again.")
    log_path.write_text(json.dumps({
        **this_run,
        "versions": _versions(),
        "image_dir": str(image_dir),
        "experiment_name": experiment_name,
        "n_images": len(files),
        "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    expected = core.EXPECTED_SHAPE[settings.resolution]
    n_skipped = 0
    for fn in tqdm(files, desc="Images"):
        stem = fn.stem
        done_marker = per_image_dir / f"{stem}__images.csv"
        if skip_existing and done_marker.exists():
            n_skipped += 1
            continue

        img = core.load_image(fn, expected_shape=expected)
        tables, masks, checks = analyze_image(img, settings)

        meta = {"filename": stem, **core.parse_filename(stem, settings.name_format)}
        for name in TABLES:
            df = tables[name]
            if name != "images" and len(df) == 0:
                continue
            for i, (col, val) in enumerate(meta.items()):
                df.insert(i, col, val)
            if name != "images":
                df.to_csv(per_image_dir / f"{stem}__{name}.csv", index=False)
        if save_masks:
            for name, m in masks.items():
                # 16-bit opens directly in Fiji; fall back to 32-bit only if an
                # image ever has more than 65,535 objects
                dtype = np.uint16 if m.max() <= np.iinfo(np.uint16).max else np.uint32
                imwrite(mask_dir / f"{stem}_{name}.tif", m.astype(dtype),
                        compression="zlib")
        if save_checks:
            for name, rgb in checks.items():
                imwrite(check_dir / f"{stem}_check_{name}.tif", rgb, compression="zlib")
        # written last: marks this image as finished
        tables["images"].to_csv(done_marker, index=False)

    if n_skipped:
        print(f"Skipped {n_skipped} image(s) already analysed in an earlier run.")

    # --- combine (only images that are in the folder now) ---
    current = {fn.stem for fn in files}
    stale = {p.name.rsplit("__", 1)[0] for p in per_image_dir.glob("*__images.csv")} - current
    if stale:
        print(f"Ignored saved results for {len(stale)} image(s) no longer in the image folder "
              "(e.g. renamed or deleted).")
    combined = {}
    for name in TABLES:
        parts = [pd.read_csv(p, dtype={c: str for c in core.name_fields(settings.name_format)})
                 for p in natural_sort_csvs(per_image_dir, name)
                 if p.name.rsplit("__", 1)[0] in current]
        out_name = settings.compartment_name if name == "compartment" else name
        out_path = csv_dir / f"{experiment_name}_{out_name}.csv"
        if parts:
            combined[out_name] = pd.concat(parts, ignore_index=True)
            combined[out_name].to_csv(out_path, index=False)
        elif out_path.exists():
            out_path.unlink()  # don't leave an old combined file behind

    n_nuc = int(combined["images"]["n_nuclei"].sum()) if "images" in combined else 0
    print(f"Done: {len(files)} images, {n_nuc} nuclei. CSVs in {csv_dir}")
    return combined


def natural_sort_csvs(folder: Path, table: str):
    return core.natural_sort(folder.glob(f"*__{table}.csv"))
