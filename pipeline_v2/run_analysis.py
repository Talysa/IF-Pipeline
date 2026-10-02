"""Run the IF analysis on one folder of TIFs (local version of the notebook).

Edit the settings below, then run:  python run_analysis.py
"""

from pathlib import Path

import pipeline_utils as P

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
IMAGE_DIR = Path(
    "/Users/talysaviera/Desktop/XRN2-Mutant-IF-Images/2026-09-03-NCL-TimeCourse/TIF"
)
OUTPUT_DIR = None          # None = save check/, masks/, csvs/ inside IMAGE_DIR

CELLPOSE_MODEL = "cpsam_v2"  # or a full path to a custom model

settings = P.Settings(
    resolution="512px",          # "512px" or "1024px"
    # how file names split into CSV columns ("MAX_" prefixes are ignored)
    name_format="date-antibody_1-antibody_2-treatment-timepoint-rep-image_number",
    dapi_channel=2,
    channels={"NCL": 1, "BP53": 0},  # every non-DAPI channel: name -> index

    foci_channels=["BP53"],      # [] for no foci
    compartment_channel="NCL",   # marker outlining a compartment; None to skip
    compartment_name="nucleoli", # e.g. "nucleoli" or "speckles"; used in file/column names
    # compartment_min_area_um2=0.75,  # smallest compartment counted (lower for speckles)
    overlap_pair=None,           # e.g. ("gH2AX", "BP53") with both as foci channels

    nucleus_diameter_um=25.0,    # None = let Cellpose use the image as-is
    # foci_overrides={"BP53": {"threshold_k": 5.0}},
)

if __name__ == "__main__":
    P.load_model(CELLPOSE_MODEL)
    P.run_folder(IMAGE_DIR, settings, output_dir=OUTPUT_DIR)
