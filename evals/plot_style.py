"""Shared plotting style and layout configuration for NeurIPS camera-ready figures."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns

# ---- NeurIPS Page & Figure Dimensions (in inches) ----
# Text width for NeurIPS is 5.5 inches (33 picas / 396 pt).
# Setting figure width to 5.5 ensures 1:1 scale in LaTeX when included via \includegraphics[width=\linewidth]{...}
NEURIPS_TEXTWIDTH_INCHES = 5.5
NEURIPS_PAGE_HEIGHT_INCHES = 9.0
DEFAULT_FIG_HEIGHT_INCHES = 2.55
DEFAULT_DPI = 300

# ---- Typography & rcParams ----
NEURIPS_RCPARAMS = {
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",  # Matches Times Roman font for math symbols ($\Delta$, $\leq$, etc.)
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelweight": "normal",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
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
}

# Standard reusable element kwargs
SEPARATOR_LINE_KW = {"color": "#e8e8e8", "linestyle": ":", "linewidth": 0.6, "zorder": 1}
ZERO_LINE_KW = {"color": "#222222", "linestyle": "-", "linewidth": 0.6, "zorder": 4}
ERRORBAR_KW = {"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#222222"}


def apply_neurips_style(style: str = "ticks", context: str = "paper"):
    """Applies NeurIPS publication theme and typography to matplotlib."""
    sns.set_theme(style=style, context=context)
    plt.rcParams.update(NEURIPS_RCPARAMS)


# ---- Colorblind-Safe Color Palette (Okabe-Ito) ----
# Standard Okabe-Ito palette (NeurIPS/ICML standard, high contrast, clean greyscale reproduction)
OKABE_ITO_PALETTE = {
    "sky_blue": "#56B4E9",
    "blue": "#0072B2",
    "bluish_green": "#009E73",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "orange": "#E69F00",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Default 4-color subset with maximum distinct hues for error-handling methods
COLORS = [
    OKABE_ITO_PALETTE["sky_blue"],
    OKABE_ITO_PALETTE["reddish_purple"],
    OKABE_ITO_PALETTE["bluish_green"],
    OKABE_ITO_PALETTE["vermilion"],
]

# ---- Modality Styles ----
MODALITY_STYLES = {
    "Structured Rollout": {"hatch": None, "alpha": 0.95, "prefix": "S"},
    "Natural Language Rollout": {"hatch": "///", "alpha": 0.88, "prefix": "NL"},
}

# ---- Error Handling Methods ----
METHODS = {
    "probability_0": "Non-occurrence ($y_t=0$)",
    "probability_0_5": "Uninformative ($y_t=0.5$)",
    "probability_prevalence": "Training Prevalence",
    "probability_drop_trajectory": "Drop Trajectories",
}

# ---- Standardized Multi-line Task Labels (Ensuring uniform vertical baselines) ----
TASK_MAP = {
    "ed_72hr_reattendance": "ED 72-Hour\nReattendance",
    "ed_to_hospital_admission": "ED to Hosp.\nAdmission",
    "hospital_length_of_stay_leq_7_days": "Hospital LOS\n$\\leq$ 7 Days",
    "hospital_post_discharge_30_day_mortality": "30-Day\nMortality",
    "in_hospital_mortality": "In-Hospital\nMortality",
}


def format_task_label(task: str) -> str:
    """Formats a task name into a standard display label."""
    if task in TASK_MAP:
        return TASK_MAP[task]

    def _process_word(w: str) -> str:
        return "ED" if w.lower() == "ed" else ("LOS" if w.lower() == "los" else (w.title() if w.islower() else w))

    return " ".join(_process_word(w) for w in task.replace("_", " ").split())


# ---- Figure Saving Utility ----
def save_figure(
    out_dir: Path,
    stem: str,
    formats: Sequence[str] = ("pdf", "png"),
    pad_inches: float = 0.02,
    dpi: int = DEFAULT_DPI,
):
    """Saves the current matplotlib figure in multiple formats with tight bounding box."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = out_dir / f"{stem}.{fmt}"
        plt.savefig(target, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
