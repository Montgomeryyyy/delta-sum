import argparse
import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from transformers import AutoTokenizer

from mimic_struct_predict.evals.plot_style import (
    NEURIPS_TEXTWIDTH_INCHES,
    TIMELINE_PALETTE,
    apply_neurips_style,
    save_figure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

apply_neurips_style()


def get_training_runtime(run_dir: Path) -> float:
    """Extracts total training runtime (in seconds) from TensorBoard logs."""
    tfevents = list(run_dir.glob("runs/**/events.out.tfevents*"))
    if not tfevents:
        raise FileNotFoundError(f"Could not find TensorBoard event logs in {run_dir}")

    ea = event_accumulator.EventAccumulator(str(tfevents[0]))
    ea.Reload()
    times = []
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        if evs:
            times.extend([evs[0].wall_time, evs[-1].wall_time])

    if not times:
        raise ValueError(f"No scalar events found in TensorBoard logs for {run_dir}")

    return float(max(times) - min(times))


def get_eval_loss_history(run_dirs: list[Path]) -> list[tuple[float, float]]:
    """Extracts (epoch, eval_loss) points directly from TensorBoard logs."""
    eval_dict: dict[float, float] = {}

    for run_dir in run_dirs:
        tfevents = list(run_dir.glob("runs/**/events.out.tfevents*"))
        if not tfevents:
            raise FileNotFoundError(f"Could not find TensorBoard event logs in {run_dir}")

        ea = event_accumulator.EventAccumulator(str(tfevents[0]), size_guidance={"scalars": 0})
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])
        if "eval/loss" not in scalars or "train/epoch" not in scalars:
            continue

        train_epoch = {e.step: e.value for e in ea.Scalars("train/epoch")}
        max_step, max_epoch = max(train_epoch.keys()), max(train_epoch.values())
        for e in ea.Scalars("eval/loss"):
            epoch = train_epoch.get(e.step, (e.step / max_step) * max_epoch)
            eval_dict[epoch] = float(e.value)

    if not eval_dict:
        raise FileNotFoundError(f"Could not find eval/loss history in {run_dirs}")

    return sorted(eval_dict.items())


def plot_loss_curves(struct_dirs: list[Path], nl_dirs: list[Path], output_dir: Path) -> None:
    """Plots side-by-side validation loss curves for Structured and NL pretraining."""
    fig, (ax_s, ax_nl) = plt.subplots(1, 2, figsize=(NEURIPS_TEXTWIDTH_INCHES, 2.0), layout="constrained")

    configs = [
        (ax_s, struct_dirs, TIMELINE_PALETTE["Structured"], "(a) Structured", np.linspace(0.4, 2.4, 5), "%.1f"),
        (
            ax_nl,
            nl_dirs,
            TIMELINE_PALETTE["Natural Language"],
            "(b) Natural Language",
            np.linspace(0.24, 0.32, 5),
            "%.2f",
        ),
    ]

    for ax, dirs, color, title, y_ticks, fmt in configs:
        epochs, losses = zip(*get_eval_loss_history(dirs))
        ax.plot(epochs, losses, color=color, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_xlim(0, max(epochs))
        ax.set_xticks(range(int(max(epochs)) + 1))
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks[0], y_ticks[-1])
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))

    sns.despine(fig=fig, trim=False)
    save_figure(output_dir, "pretraining_loss_curves")
    plt.close(fig)
    log.info(f"Saved pretraining loss curves to {output_dir / 'pretraining_loss_curves.png'} (and .pdf)")


def format_duration(seconds: float) -> str:
    """Formats duration in hours and total seconds."""
    hours = seconds / 3600
    td = str(datetime.timedelta(seconds=int(seconds)))
    return f"{hours:.2f} hours ({td}, {seconds:,.1f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tokenizer details, total training times, and loss curves.")
    parser.add_argument("--struct_model_dir", type=Path, default=Path(os.environ["STRUCT_MODEL_DIR"]))
    parser.add_argument("--nl_model_dir", type=Path, default=Path(os.environ["NL_MODEL_DIR"]))
    parser.add_argument(
        "--struct_training_dirs",
        type=Path,
        nargs="+",
        default=[Path(os.environ["STRUCT_MODEL_DIR"]).parent],
    )
    parser.add_argument(
        "--nl_training_dirs",
        type=Path,
        nargs="+",
        default=[
            Path(os.environ["NL_MODEL_DIR"]).parent,
            # HACK: To append split model training run together
            Path(
                str(Path(os.environ["NATURAL_LANGUAGE_MODEL_DIR"]) / "pretraining" / "20260817_225728"),
            ),
        ],
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ["EVAL_PLOTS_DIR"]) / "loss_curves",
    )
    args = parser.parse_args()

    print(f"STRUCT_MODEL_DIR tokenizer length: {len(AutoTokenizer.from_pretrained(args.struct_model_dir))}")
    print(f"NL_MODEL_DIR tokenizer length: {len(AutoTokenizer.from_pretrained(args.nl_model_dir))}")

    for label, dirs in [
        ("Structured Model", args.struct_training_dirs),
        ("Natural Language Model", args.nl_training_dirs),
    ]:
        runtimes = {d: get_training_runtime(d) for d in dirs}
        print(f"\n{label} Total Training Time: {format_duration(sum(runtimes.values()))}")
        for d, runtime in runtimes.items():
            print(f"  - {d.name}: {format_duration(runtime)}")

    plot_loss_curves(args.struct_training_dirs, args.nl_training_dirs, args.output_dir)


if __name__ == "__main__":
    main()
