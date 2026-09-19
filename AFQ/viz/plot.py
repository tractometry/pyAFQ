import math

import matplotlib.pyplot as plt
import pandas as pd

from AFQ.viz.utils import COLOR_DICT, display_string

__all__ = ["visualize_tract_profiles"]


def _plot_tract(ax, df, tract_name, metric, color, label=None):
    sub = df[df["tractID"] == tract_name].sort_values("nodeID")
    if sub.empty:
        return
    ax.plot(
        sub["nodeID"],
        sub[metric],
        color=color,
        linewidth=1.8,
        label=label or tract_name,
    )


def _split_hemisphere(tract_id):
    for prefix, hemi in (("Left ", "Left"), ("Right ", "Right")):
        if tract_id.startswith(prefix):
            return tract_id[len(prefix) :], hemi
    for suffix, hemi in (("_L", "Left"), ("_R", "Right")):
        if tract_id.endswith(suffix):
            return tract_id[: -len(suffix)], hemi
    return None, None


def visualize_tract_profiles(
    tract_profiles,
    scalar="dti_fa",
    file_name=None,
    fontsize=14,
):
    """
    Visualize all tract profiles for a scalar in one plot

    Parameters
    ----------
    tract_profiles : string
        Path to CSV containing tract_profiles.

    scalar : string, optional
       Scalar to use in plots. Default: "dti_fa".

    file_name : string, optional
        File name to save figure to if not None. Default: None

    fontsize : int, optional
        Font size for figure. Default: 14

    Returns
    -------
        Matplotlib figure and axes.
    """
    df = pd.read_csv(tract_profiles)

    bilateral = {}
    callosal = []
    for tt in df["tractID"].unique():
        base, hemi = _split_hemisphere(tt)
        if base is None:
            callosal.append(tt)
        else:
            bilateral.setdefault(base, {})[hemi] = tt

    bilateral_bases = sorted(bilateral)
    callosal = sorted(callosal)

    n_bilateral = len(bilateral_bases)
    n_panels = n_bilateral + len(callosal)

    n_cols = math.ceil(math.sqrt(n_panels))
    n_rows = math.ceil(n_panels / n_cols)

    fig1, axes1 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    flat_axes = axes1.ravel()

    def _style_axis(ax, ylabel):
        ax.set_ylabel(ylabel, fontsize=fontsize - 2, labelpad=4)
        ax.set_xlabel("Node", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=fontsize, loc="best", frameon=False)

    default_colors = {"Left": "steelblue", "Right": "darkorange"}

    for idx, base in enumerate(bilateral_bases):
        ax = flat_axes[idx]
        for hemi in ("Left", "Right"):
            full_name = bilateral[base].get(hemi)
            if full_name is None:
                continue
            _plot_tract(
                ax,
                df,
                full_name,
                scalar,
                COLOR_DICT.get(full_name, default_colors[hemi]),
                hemi,
            )
        _style_axis(ax, base + " " + display_string(scalar))

    for ii, tract in enumerate(callosal):
        ax = flat_axes[n_bilateral + ii]
        sub = df[df["tractID"] == tract].sort_values("nodeID")
        ax.plot(
            sub["nodeID"],
            sub[scalar],
            color=COLOR_DICT.get(tract, "gray"),
            linewidth=1.8,
            label=tract.replace("Callosum ", ""),
        )
        _style_axis(ax, tract + " " + display_string(scalar))

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig1.suptitle(display_string(scalar), fontsize=fontsize + 2, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    if file_name is not None:
        fig1.savefig(file_name, dpi=300, bbox_inches="tight")

    return fig1, axes1
