"""Head position and direction pipeline for SLEAP tracking.

Adapted from HeadCentroid_Directory_YJL.ipynb. Loads the h5 SLEAP output,
cleans the pose tracks (confidence filter -> rolling median -> interpolate),
and derives head position (ears midpoint) and head direction (forward-vector
angle), storing both as extra data variables on the movement dataset. Saves
diagnostic plots and the processed dataset, then returns a pynapple TsdFrame
of head position and direction via ``movement_to_nap``.
"""
# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import xarray as xr
from movement.io import load_dataset
from movement.roi import load_rois
from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    rolling_filter,
)
from movement.kinematics import compute_forward_vector_angle

from movement_to_pynapple import movement_to_nap

FPS = 25
WORKING_DIR = Path(__file__).parent
DATA_DIR = WORKING_DIR / "data" / "SLEAP_data(behavior)"
H5_FILE = DATA_DIR / "T1_1D_04032026.h5"
OUT_DS_FILE = DATA_DIR / "T1_1D_04032026_processed_movement.nc"
OUT_NAP_FILE = DATA_DIR / "T1_1D_04032026_processed_pynapple.npz"
ROIS_FILE = WORKING_DIR / "data" / "goal_roi.geojson"
FRAME_FILE = WORKING_DIR / "vlcsnap-2026-08-21-15h13m03s667.png"


def clean(
    ds: xr.Dataset,
    confidence_threshold: float = 0.75,
    max_gap: float = 5,
    rolling_window: float = 7,
    rolling_min_periods: int = 2,
) -> xr.Dataset:
    """Filter low-confidence points, interpolate gaps, rolling-median smooth."""
    pos = filter_by_confidence(
        ds.position, ds.confidence,
        threshold=confidence_threshold, print_report=False,
    )
    pos = rolling_filter(
        pos,
        window=rolling_window,
        min_periods=rolling_min_periods,
        statistic="median",
        print_report=False,
    )
    pos = interpolate_over_time(
        pos, max_gap=max_gap, print_report=False
    )
    ds_clean = ds.copy()
    ds_clean.update({"position": pos})
    return ds_clean


def head_centroid(ds: xr.Dataset) -> xr.DataArray:
    """Head centroid: midpoint between the two ears."""
    return ds.position.sel(keypoint=["left_ear", "right_ear"]).mean(
        dim="keypoint"
    )


def plot_centroid_raw_vs_clean(ds_raw, ds_clean, roi_file=ROIS_FILE):
    """Raw and cleaned head-centroid trajectory (x-y) side by side."""
    raw = head_centroid(ds_raw).squeeze()
    cln = head_centroid(ds_clean).squeeze()
    frame = plt.imread(FRAME_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, da, title in [(axes[0], raw, "raw"), (axes[1], cln, "cleaned")]:
        ax.imshow(frame, zorder=0)  # origin upper -> image coordinates
        sc = ax.scatter(da.sel(space="x"), da.sel(space="y"),
                        c=da.time, cmap="viridis", s=4, zorder=1)
        ax.set_aspect("equal")
        ax.set_xlabel("x (pixels)")
        ax.set_title(f"Head centroid: {title}")
        if roi_file is not None:
            rois = load_rois(roi_file)
            for roi in rois:
                roi.plot(
                    ax=ax, facecolor="none", edgecolor="red", lw=1, label=roi.name
                )
            ax.legend(loc="upper left")


    axes[0].set_xlim(500, 2000)
    axes[0].set_ylim(1500, 250)
    axes[0].set_ylabel("y (pixels)")
    fig.colorbar(sc, ax=axes, label="time (s)", shrink=0.6)
    return fig

def plot_head_angle(head_direction, bin_width_deg=15):
    """Polar histogram + time evolution of head-direction angle."""
    import numpy as np

    n_bins = int(360 / bin_width_deg)
    fig, (ax_hist, ax_time) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"}
    )

    head_direction.plot.hist(
        bins=np.linspace(-np.pi, np.pi, n_bins + 1), ax=ax_hist, density=True
    )
    ax_hist.set_theta_direction(-1)  # clockwise
    ax_hist.set_xlabel("")
    ax_hist.set_title("Head direction distribution")

    # theta = angle, radius = time, coloured by time
    ax_time.scatter(
        head_direction, head_direction.time,
        c=head_direction.time, cmap="viridis", s=6,
    )
    ax_time.set_theta_direction(-1)
    ax_time.set_title("Head direction over time")
    return fig

# %%

def main(save_plots: bool = True, save_ds: bool = True, **clean_kwargs):
    """Load, clean, derive head metrics, and convert to pynapple.

    Loads the SLEAP h5 tracks, cleans them, adds ``head_centroid`` and
    ``head_direction`` as data variables, optionally saves the dataset to
    netCDF, and returns the pynapple conversion of the result.

    Parameters
    ----------
    save_plots : bool, default True
        If True, save the head-centroid and head-angle diagnostic plots as
        PNG files next to the script.
    save_ds : bool, default True
        If True, write the processed dataset to ``OUT_DS_FILE`` as netCDF.
    **clean_kwargs
        Keyword arguments forwarded to :func:`clean` (e.g.
        ``confidence_threshold``, ``max_gap``, ``rolling_window``).

    Returns
    -------
    pynapple.TsdFrame
        Head position and direction on a shared time axis, with columns
        ``head_x``, ``head_y``, ``head_direction``.
    """
    ds_raw = load_dataset(H5_FILE, fps=FPS)
    ds = clean(ds_raw, **clean_kwargs)

    ds["head_position"] = head_centroid(ds)

    # Head direction: forward-vector angle (radians) from the ears line.
    ds["head_direction"] = compute_forward_vector_angle(
        ds.position,
        left_keypoint="left_ear",
        right_keypoint="right_ear",
        reference_vector=(1, 0),
        camera_view="top_down",
        in_degrees=False,
    )

    if save_plots:
        plot_centroid_raw_vs_clean(ds_raw, ds).savefig(
            WORKING_DIR / "head_centroid.png", dpi=150, bbox_inches="tight"
        )
        plot_head_angle(ds["head_direction"].squeeze()).savefig(
            WORKING_DIR / "head_angle.png", dpi=150, bbox_inches="tight"
        )
        plt.close("all")

    if save_ds:
        ds.to_netcdf(OUT_DS_FILE)
        print(f"Saved movement dataset: {OUT_DS_FILE}")

    nap_data = movement_to_nap(ds)
    pos, angle = nap_data["head_position"], nap_data["head_direction"]
    # One TsdFrame (shared time axis) prints as a clean labelled table.
    return nap.TsdFrame(
        t=pos.index.values,
        d=np.column_stack([pos.values, angle.values]),
        columns=["head_x", "head_y", "head_direction"],
    )

# %%

if __name__ == "__main__":
    nap_tsdf = main(save_plots=True, save_ds=True)
    nap_tsdf.save(OUT_NAP_FILE)
    print(f"Saved pynaple TsdFrame: {OUT_NAP_FILE}")
    print(nap_tsdf)


