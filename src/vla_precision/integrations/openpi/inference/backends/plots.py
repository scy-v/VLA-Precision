"""Plot recorded native-inference episodes."""

import argparse
import textwrap
from pathlib import Path
from typing import Any

import yaml


def _load_yaml_docs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    docs = []
    for part in text.split(" ------------------------------------------------------------"):
        part = part.strip()
        if not part:
            continue
        data = yaml.safe_load(part)
        if isinstance(data, dict):
            docs.append(data)
    return docs


def load_episode_summary(log_path: Path) -> dict[str, Any]:
    docs = _load_yaml_docs(log_path)
    for doc in reversed(docs):
        summary = doc.get("episode_summary")
        if isinstance(summary, dict):
            return summary
    raise ValueError(f"No episode_summary found in {log_path}")


def save_episode_plot(summary: dict[str, Any], log_path: Path | None = None, output_path: Path | None = None) -> str | None:
    episodes = summary.get("episodes", [])
    if not episodes:
        return None

    timed = [ep for ep in episodes if ep.get("duration_sec") is not None]
    if not timed:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if output_path is None:
        base_dir = Path(__file__).resolve().parent / "logs" / "ep_img"
        if log_path is None:
            stem = "episode_time"
        else:
            resolved_log = log_path.resolve() if log_path.exists() else log_path
            stem = f"{resolved_log.stem}_episode_time"
        output_path = base_dir / f"{stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_all = [ep["episode_index"] for ep in timed]
    y_all = [ep["duration_sec"] for ep in timed]
    done = [ep for ep in timed if ep.get("completed")]
    other = [ep for ep in timed if not ep.get("completed")]
    x_done = [ep["episode_index"] for ep in done]
    y_done = [ep["duration_sec"] for ep in done]
    x_other = [ep["episode_index"] for ep in other]
    y_other = [ep["duration_sec"] for ep in other]
    total = summary.get("total_episodes", len(episodes))
    completed = summary.get("completed_episodes", len(done))
    reset_count = max(total - completed, len(other))
    success_rate = summary.get("success_rate")
    if success_rate is None:
        success_rate = completed / total if total else 0.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.2,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.75,
    })
    plot_width = min(max(8.8, len(timed) * 0.12), 20.0)
    fig, ax = plt.subplots(figsize=(plot_width, 3.8), dpi=260)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x_all, y_all, color="0.72", linewidth=0.95, zorder=1)
    if x_done:
        ax.scatter(
            x_done,
            y_done,
            s=58,
            marker="o",
            facecolor="#2a9d8f",
            edgecolor="#1f5f5b",
            linewidth=0.85,
            alpha=0.96,
            label=f"Done ({completed}/{total})",
            zorder=3,
        )
    if x_other:
        ax.scatter(
            x_other,
            y_other,
            s=34,
            marker="D",
            facecolor="white",
            edgecolor="#b22222",
            linewidth=1.0,
            label=f"Reset ({reset_count}/{total})",
            zorder=4,
        )

    avg = summary.get("average_completed_duration_sec", 0.0)
    if x_done and avg:
        ax.axhline(
            avg,
            color="#b22222",
            linestyle="--",
            linewidth=0.95,
            label=f"Average time ({avg:.1f}s)",
            zorder=2,
        )

    description = summary.get("description")
    task_description = f"Task: {description}" if description else ""
    wrapped_description = "\n".join(textwrap.wrap(task_description, width=58)) if task_description else ""
    ax.add_line(Line2D([], [], color="black", linewidth=0.95, label=f"Success rate ({success_rate:.1%})"))

    action_horizon = summary.get("action_horizon")
    ax.set_title("Episode Time", pad=36 if wrapped_description else 10, fontweight="semibold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Time (s)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", color="0.88", linewidth=0.55)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8, width=0.7)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(1.26, 0.66),
        ncol=1,
        frameon=True,
        fancybox=True,
        edgecolor="#d4d8dc",
        facecolor="#fbfcfd",
        framealpha=1.0,
        borderpad=0.55,
        handlelength=1.45,
        handletextpad=0.58,
        labelspacing=0.62,
    )
    legend.get_frame().set_linewidth(0.75)

    fig.tight_layout(rect=(0, 0, 0.78, 1))

    def _overlay_colored_suffix(text_obj, prefix, suffix, color, background="#fbfcfd"):
        renderer = fig.canvas.get_renderer()
        prop = text_obj.get_fontproperties()
        bbox = text_obj.get_window_extent(renderer=renderer)
        prefix_width, _, _ = renderer.get_text_width_height_descent(prefix, prop, ismath=False)
        x_fig, y_fig = fig.transFigure.inverted().transform((bbox.x0 + prefix_width, bbox.y0 + bbox.height / 2))
        fig.text(
            x_fig,
            y_fig,
            suffix,
            ha="left",
            va="center",
            fontsize=text_obj.get_fontsize(),
            fontproperties=prop,
            color=color,
            bbox={"facecolor": background, "edgecolor": "none", "pad": 0.0},
            zorder=20,
        )

    fig.canvas.draw()
    title_suffix_width = 0.0
    if action_horizon is not None:
        renderer = fig.canvas.get_renderer()
        title_suffix = f" (action_horizon={action_horizon})"
        title_suffix_width, _, _ = renderer.get_text_width_height_descent(
            title_suffix,
            ax.title.get_fontproperties(),
            ismath=False,
        )
        _overlay_colored_suffix(
            ax.title,
            "Episode Time",
            title_suffix,
            "#5287bd",
            background="white",
        )

    if wrapped_description:
        renderer = fig.canvas.get_renderer()
        title_bbox = ax.title.get_window_extent(renderer=renderer)
        title_center_x = title_bbox.x0 + (title_bbox.width + title_suffix_width) / 2
        x_fig, y_fig = fig.transFigure.inverted().transform((title_center_x, title_bbox.y0 - 40))
        fig.text(
            x_fig,
            y_fig,
            wrapped_description,
            ha="center",
            va="top",
            fontsize=11,
            fontfamily="Liberation Sans",
            fontweight="semibold",
            color="#AF3A16",
            zorder=20,
        )

    suffix_specs = []
    if x_done:
        suffix_specs.append(("Done ", f"({completed}/{total})", "#37A19A"))
    if x_other:
        suffix_specs.append(("Reset ", f"({reset_count}/{total})", "#b22222"))
    if x_done and avg:
        suffix_specs.append(("Average time ", f"({avg:.1f}s)", "#8a3ffc"))
    suffix_specs.append(("Success rate ", f"({success_rate:.1%})", "#DA28C2"))
    for text_obj, (prefix, suffix, color) in zip(legend.get_texts(), suffix_specs):
        _overlay_colored_suffix(text_obj, prefix, suffix, color)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Plot inference episode duration from an OpenPI log.")
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "logs" / "latest.yaml",
        help="Path to a log yaml file. Defaults to scripts/logs/latest.yaml.",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Optional output image path.")
    args = parser.parse_args()

    summary = load_episode_summary(args.log_path)
    output = args.output
    if output is None:
        base_dir = Path(__file__).resolve().parent / "logs" / "ep_img"
        resolved_log = args.log_path.resolve() if args.log_path.exists() else args.log_path
        output = base_dir / f"vis_{resolved_log.stem}_episode_time.png"
    output = save_episode_plot(summary, log_path=args.log_path, output_path=output)
    if output is None:
        print("No timed episode data found; no plot generated.")
    else:
        print(f"Episode plot saved to: {output}")


if __name__ == "__main__":
    main()
