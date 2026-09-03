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

    callosal = sorted([t for t in df["tractID"].unique() if t.startswith("Callosum")])

    bilateral_bases = sorted(
        {
            t.replace("Left ", "").replace("Right ", "")
            for t in df["tractID"].unique()
            if t.startswith("Left ") or t.startswith("Right ")
        }
    )

    n_bilateral = len(bilateral_bases)
    n_callosal = len(callosal)
    n_panels = n_bilateral + n_callosal

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

    for idx, base in enumerate(bilateral_bases):
        ax = flat_axes[idx]
        _plot_tract(
            ax,
            df,
            f"Left {base}",
            scalar,
            COLOR_DICT.get(f"Left {base}", "steelblue"),
            "Left",
        )
        _plot_tract(
            ax,
            df,
            f"Right {base}",
            scalar,
            COLOR_DICT.get(f"Right {base}", "darkorange"),
            "Right",
        )

        ax.set_ylabel(
            base + " " + display_string(scalar), fontsize=fontsize - 2, labelpad=4
        )
        ax.set_xlabel("Node", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=fontsize, loc="best", frameon=False)

    callosal_colors = {name: COLOR_DICT.get(name, "gray") for name in callosal}

    for ii, tract in enumerate(callosal):
        ax = flat_axes[n_bilateral + ii]
        sub = df[df["tractID"] == tract].sort_values("nodeID")
        short = tract.replace("Callosum ", "")
        ax.plot(
            sub["nodeID"],
            sub[scalar],
            color=callosal_colors[tract],
            linewidth=1.8,
            label=short,
        )

        ax.set_ylabel(
            tract + " " + display_string(scalar), fontsize=fontsize - 2, labelpad=4
        )
        ax.set_xlabel("Node", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=fontsize, loc="best", frameon=False)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig1.suptitle(display_string(scalar), fontsize=fontsize + 2, fontweight="bold")
    fig1.tight_layout()

    if file_name is not None:
        fig1.savefig(file_name, dpi=300, bbox_inches="tight")

    return fig1, axes1
