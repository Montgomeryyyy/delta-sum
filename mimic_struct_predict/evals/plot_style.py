"""Shared plotting style and layout configuration for NeurIPS camera-ready figures."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ---- NeurIPS Page & Figure Dimensions (in inches) ----
# Text width for NeurIPS is 5.5 inches (33 picas / 396 pt).
# Setting figure width to 5.5 ensures 1:1 scale in LaTeX when included via \includegraphics[width=\linewidth]{...}
NEURIPS_TEXTWIDTH_INCHES = 5.5
NEURIPS_PAGE_HEIGHT_INCHES = 9.0
DEFAULT_FIG_HEIGHT_INCHES = 2.7
DEFAULT_DPI = 300

# ---- Typography & rcParams ----
NEURIPS_RCPARAMS = {
    # Typography
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "mathtext.fontset": "stix",  # Matches Times Roman font for math symbols ($\Delta$, $\leq$, etc.)
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelweight": "normal",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "legend.fontsize": 6.8,
    # Legend layout defaults
    "legend.frameon": False,
    "legend.columnspacing": 0.6,
    "legend.handletextpad": 0.3,
    "legend.handlelength": 1.1,
    "legend.handleheight": 0.7,
    "legend.borderaxespad": 0.2,
    # Grid defaults
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "grid.color": "#888888",
    # Patch / Bar defaults
    "patch.edgecolor": "#222222",
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "lines.linewidth": 0.6,
    "axes.linewidth": 0.6,
    # Error bars
    "errorbar.capsize": 1.5,
    # Resolution and Saving
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Standard reusable element kwargs
SEPARATOR_LINE_KW = {"color": "#e8e8e8", "linestyle": ":", "linewidth": 0.6, "zorder": 1}
ZERO_LINE_KW = {"color": "#222222", "linestyle": "-", "linewidth": 0.6, "zorder": 4}
ERRORBAR_KW = {"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#222222"}


def apply_neurips_style(style: str = "ticks", context: str = "paper") -> None:
    """Applies NeurIPS publication theme and typography to matplotlib."""
    sns.set_theme(style=style, context=context, palette="tab10")
    plt.rcParams.update(NEURIPS_RCPARAMS)
    # Suppress verbose font subsetting and layout logs from fontTools / matplotlib
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ---- Matplotlib Tab10 Color Palette ----
TAB10 = plt.get_cmap("tab10")
TAB10_COLORS = list(TAB10.colors)

# ---- Distinct Palettes Per Figure (Tab10 Colormap) ----
# Figure 1: Format Error Reasons (evals/error_rates.py)
ERROR_RATES_PALETTE = {
    "Category Error": TAB10(0),
    "Date Error": TAB10(1),
    "Time Error": TAB10(2),
    "Max New Tokens Reached": TAB10(3),
}

# Figure 2: Trajectory Lengths Modalities & Threshold (evals/gt_trajectory_lengths.py)
TIMELINE_PALETTE = {
    "Structured": TAB10(4),
    "Natural Language": TAB10(5),
    "Structured (GT)": TAB10(4),
    "Natural Language (GT)": TAB10(5),
    "threshold_line": TAB10(3),  # Red
}

# Figure 3: Error-Handling Imputation Methods (evals/rollout_delta.py)
METHODS_PALETTE = {
    "probability_0": TAB10(6),
    "probability_0_5": TAB10(7),
    "probability_prevalence": TAB10(8),
    "probability_drop_trajectory": TAB10(9),
}

# Default COLORS alias for backwards compatibility
COLORS = list(METHODS_PALETTE.values())

# ---- Modality Styles ----
MODALITY_STYLES: dict[str, dict[str, Any]] = {
    "Structured Rollout": {"hatch": None, "alpha": 0.95, "prefix": "S", "label": "Structured"},
    "Natural Language Rollout": {"hatch": "///", "alpha": 0.88, "prefix": "NL", "label": "Natural Language"},
}

# ---- Error Handling Methods ----
METHODS = {
    "probability_0": "Non-occurrence ($y_t=0$)",
    "probability_0_5": "Uninformative ($y_t=0.5$)",
    "probability_prevalence": "Training Prevalence",
    "probability_drop_trajectory": "Drop Trajectories",
}

# ---- Standardized Multi-line Task Labels (Ensuring uniform vertical baselines) ----
# Order matches prediction time in a patient stay
TASK_ORDER = [
    "ed_to_hospital_admission",
    "ed_72hr_reattendance",
    "hospital_length_of_stay_leq_7_days",
    "in_hospital_mortality",
    "hospital_post_discharge_30_day_mortality",
]

TASK_MAP = {
    "ed_to_hospital_admission": "ED to Hosp.\nAdmission",
    "ed_72hr_reattendance": "ED 72-Hour\nReattendance",
    "hospital_length_of_stay_leq_7_days": "Hospital LOS\n$\\leq$ 7 Days",
    "in_hospital_mortality": "In-Hospital\nMortality",
    "hospital_post_discharge_30_day_mortality": "30-Day\nMortality",
}


def sort_tasks(tasks: list[str]) -> list[str]:
    """Sorts tasks according to the canonical publication order."""
    order_dict = {t: i for i, t in enumerate(TASK_ORDER)}
    return sorted(tasks, key=lambda t: order_dict.get(t, 999))


def format_task_label(task: str) -> str:
    """Formats a task name into a standard multi-line display label for plots."""
    if task in TASK_MAP:
        return TASK_MAP[task]

    def _process_word(w: str) -> str:
        return "ED" if w.lower() == "ed" else ("LOS" if w.lower() == "los" else (w.title() if w.islower() else w))

    return " ".join(_process_word(w) for w in task.replace("_", " ").split())


def format_task_title(task: str) -> str:
    """Formats a task name into a standard single-line display title (TASK_MAP without newlines)."""
    return format_task_label(task).replace("\n", " ")


def get_modality_legend_handles(facecolor: str = "white", include_prefix: bool = True) -> list[Patch]:
    """Generates standard modality legend patches (S solid, NL hatched)."""
    return [
        Patch(
            facecolor=facecolor,
            hatch=MODALITY_STYLES[mod]["hatch"],
            edgecolor="#222222",
            linewidth=0.5,
            label=f"{MODALITY_STYLES[mod]['prefix']}: {mod}" if include_prefix else mod,
            alpha=MODALITY_STYLES[mod]["alpha"],
        )
        for mod in ["Structured Rollout", "Natural Language Rollout"]
    ]


# ---- Figure Saving Utility ----
def save_figure(
    out_dir: Path,
    stem: str,
    pad_inches: float = 0.02,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Saves the current matplotlib figure as high-resolution PNG and vector PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad_inches)
