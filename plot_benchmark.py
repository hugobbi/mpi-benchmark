#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


VARIANT_LABELS = {
    "blocking": "Blocking",
    "non-blocking": "Non-blocking",
    "collective": "Collective",
}

VARIANT_COLORS = {
    "blocking": "#6c757d",
    "non-blocking": "#0077b6",
    "collective": "#2dc653",
}

METRICS = ("comm_time", "comp_time", "total_time")
PLOT_METRICS = ("comm_time", "comp_time", "total_time", "speedup", "efficiency")

plt.rcParams.update(
    {
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    }
)


@dataclass(frozen=True)
class SampleStats:
    mean: float
    std: float


@dataclass(frozen=True)
class PointStats:
    n: int
    comm_time: SampleStats
    comp_time: SampleStats
    total_time: SampleStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read MPI benchmark CSV output and generate graphs."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV file produced by the benchmark job.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Directory where the graphs will be written.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_LABELS),
        help="Plot only a specific variant (blocking, non-blocking, or collective).",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows

        expected_fields = {"variant", "n", "size", "comm_time", "comp_time", "total_time"}
        if not expected_fields.issubset({field.strip() for field in reader.fieldnames if field}):
            handle.seek(0)
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "variant,n,size,comm_time,comp_time,total_time":
                    continue

                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 6:
                    continue
                rows.append(
                    {
                        "variant": parts[0],
                        "n": parts[1],
                        "size": parts[2],
                        "comm_time": parts[3],
                        "comp_time": parts[4],
                        "total_time": parts[5],
                    }
                )
            return rows

        handle.seek(0)
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            variant = (row.get("variant") or "").strip()
            if variant not in VARIANT_LABELS:
                continue
            try:
                int((row.get("n") or "").strip())
                int((row.get("size") or "").strip())
                float((row.get("comm_time") or "").strip())
                float((row.get("comp_time") or "").strip())
                float((row.get("total_time") or "").strip())
            except ValueError:
                continue
            rows.append(
                {
                    "variant": variant,
                    "n": (row.get("n") or "").strip(),
                    "size": (row.get("size") or "").strip(),
                    "comm_time": (row.get("comm_time") or "").strip(),
                    "comp_time": (row.get("comp_time") or "").strip(),
                    "total_time": (row.get("total_time") or "").strip(),
                }
            )

    return rows


def aggregate_rows(rows: list[dict[str, str]]) -> dict[str, dict[int, list[PointStats]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, float]]] = defaultdict(list)

    for row in rows:
        key = (row["variant"], int(row["size"]), int(row["n"]))
        grouped[key].append(
            {
                "comm_time": float(row["comm_time"]),
                "comp_time": float(row["comp_time"]),
                "total_time": float(row["total_time"]),
            }
        )

    aggregated: dict[str, dict[int, list[PointStats]]] = defaultdict(lambda: defaultdict(list))

    for (variant, size, n), samples in grouped.items():
        comm_values = [sample["comm_time"] for sample in samples]
        comp_values = [sample["comp_time"] for sample in samples]
        total_values = [sample["total_time"] for sample in samples]

        def summarize(values: list[float]) -> SampleStats:
            if len(values) == 1:
                return SampleStats(mean=values[0], std=0.0)
            return SampleStats(mean=statistics.fmean(values), std=statistics.pstdev(values))

        aggregated[variant][size].append(
            PointStats(
                n=n,
                comm_time=summarize(comm_values),
                comp_time=summarize(comp_values),
                total_time=summarize(total_values),
            )
        )

    for variant_data in aggregated.values():
        for size, series in variant_data.items():
            series.sort(key=lambda item: item.n)

    return aggregated


def _speedup_and_error(base: SampleStats, current: SampleStats) -> tuple[float, float]:
    speedup = base.mean / current.mean
    if base.mean == 0.0 or current.mean == 0.0:
        return speedup, 0.0

    relative_error = 0.0
    if base.std > 0.0:
        relative_error += (base.std / base.mean) ** 2
    if current.std > 0.0:
        relative_error += (current.std / current.mean) ** 2
    return speedup, speedup * relative_error ** 0.5


def _plot_metric_lines(
    ax: plt.Axes,
    variant_data: dict[int, list[PointStats]],
    metric: str,
    color_for_size: dict[int, str],
) -> None:
    sizes = sorted(variant_data)
    for size in sizes:
        series = variant_data[size]
        xs = [point.n for point in series]

        if metric == "comm_time":
            ys = [point.comm_time.mean for point in series]
            errs = [point.comm_time.std for point in series]
            ylabel = "comm_time (s)"
        elif metric == "comp_time":
            ys = [point.comp_time.mean for point in series]
            errs = [point.comp_time.std for point in series]
            ylabel = "comp_time (s)"
        elif metric == "total_time":
            ys = [point.total_time.mean for point in series]
            errs = [point.total_time.std for point in series]
            ylabel = "total_time (s)"
        elif metric == "speedup":
            baseline = None
            if 1 in variant_data:
                baseline = {point.n: point.total_time for point in variant_data[1]}
            if baseline is None:
                continue

            xs = []
            ys = []
            errs = []
            for point in series:
                base_stats = baseline.get(point.n)
                if base_stats is None:
                    continue
                speedup, speedup_std = _speedup_and_error(base_stats, point.total_time)
                xs.append(point.n)
                ys.append(speedup)
                errs.append(speedup_std)
            ylabel = "speedup"
        elif metric == "efficiency":
            baseline = None
            if 1 in variant_data:
                baseline = {point.n: point.total_time for point in variant_data[1]}
            if baseline is None:
                continue

            xs = []
            ys = []
            errs = []
            for point in series:
                base_stats = baseline.get(point.n)
                if base_stats is None:
                    continue
                speedup, speedup_std = _speedup_and_error(base_stats, point.total_time)
                efficiency = speedup / size
                xs.append(point.n)
                ys.append(efficiency)
                errs.append(speedup_std / size)
            ylabel = "efficiency"
        else:
            continue

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                marker="o",
                linewidth=1.6,
                capsize=3,
                label=f"np={size}",
                color=color_for_size[size],
            )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Matrix size n")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Processes", fontsize=9)


def plot_variant(variant: str, variant_data: dict[int, list[PointStats]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(variant_data)
    palette = plt.get_cmap("tab10")
    color_for_size = {size: palette(index % 10) for index, size in enumerate(sizes)}

    metric_titles = {
        "comm_time": "Communication time",
        "comp_time": "Computation time",
        "total_time": "Total time",
        "speedup": "Speedup from total time",
        "efficiency": "Efficiency from total time",
    }

    metric_labels = {
        "comm_time": "comm_time",
        "comp_time": "comp_time",
        "total_time": "total_time",
        "speedup": "speedup",
        "efficiency": "efficiency",
    }

    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        fig.suptitle(f"MPI benchmark - {VARIANT_LABELS.get(variant, variant)}")

        _plot_metric_lines(ax, variant_data, metric, color_for_size)

        if metric == "efficiency":
            ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1.2, label="Ideal")

        ax.set_title(metric_titles[metric])
        ax.set_ylabel(f"{metric_labels[metric]} (s)" if metric.endswith("time") else metric_labels[metric])

        plot_name = f"{variant}_{metric}.png"
        fig.savefig(outdir / plot_name, dpi=200)
        plt.close(fig)


def plot_summary_comparison(aggregated: dict[str, dict[int, list[PointStats]]], outdir: Path) -> None:
    variants = [variant for variant in VARIANT_LABELS if variant in aggregated]
    if not variants:
        return

    all_sizes = sorted({size for variant_data in aggregated.values() for size in variant_data})
    if not all_sizes:
        return

    max_size = max(all_sizes)
    candidate_ns = {
        point.n
        for variant in variants
        for point in aggregated[variant].get(max_size, [])
    }
    if not candidate_ns:
        return

    target_n = max(candidate_ns)

    summary: dict[str, dict[str, SampleStats]] = {}
    for variant in variants:
        points = aggregated[variant].get(max_size, [])
        point = next((item for item in points if item.n == target_n), None)
        if point is None:
            continue
        summary[variant] = {
            "comm_time": point.comm_time,
            "comp_time": point.comp_time,
            "total_time": point.total_time,
        }

    if not summary:
        return

    x_positions = range(len(summary))
    labels = [VARIANT_LABELS.get(variant, variant) for variant in summary]
    colors = [VARIANT_COLORS.get(variant, "#888888") for variant in summary]

    summary_dir = outdir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "comm_time": "Communication time comparison",
        "comp_time": "Computation time comparison",
        "total_time": "Total time comparison",
    }

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        fig.suptitle(f"MPI benchmark comparison - n={target_n}, np={max_size}")

        means = [summary[variant][metric].mean for variant in summary]
        errs = [summary[variant][metric].std for variant in summary]
        bars = ax.bar(x_positions, means, yerr=errs, color=colors, capsize=5, width=0.55)
        ax.set_xticks(list(x_positions), labels)
        ax.set_ylabel(f"{metric} (s)")
        ax.set_title(metric_titles[metric])
        ax.grid(True, axis="y", alpha=0.3)

        for bar, value in zip(bars, means, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(value * 0.02, 1e-6),
                f"{value:.4f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Variant")
        fig.savefig(summary_dir / f"summary_comparison_{metric}.png", dpi=200)
        plt.close(fig)


def print_summary(aggregated: dict[str, dict[int, list[PointStats]]]) -> None:
    print("Aggregated benchmark statistics:")
    for variant in sorted(aggregated):
        print(f"  {VARIANT_LABELS.get(variant, variant)}")
        for size in sorted(aggregated[variant]):
            for point in aggregated[variant][size]:
                print(
                    f"    np={size:2d} n={point.n:4d} | "
                    f"comm={point.comm_time.mean:.6f} ± {point.comm_time.std:.6f} | "
                    f"comp={point.comp_time.mean:.6f} ± {point.comp_time.std:.6f} | "
                    f"total={point.total_time.mean:.6f} ± {point.total_time.std:.6f}"
                )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    aggregated = aggregate_rows(rows)

    if not aggregated:
        raise SystemExit("No benchmark rows were found in the input CSV.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.variant is not None:
        selected_variants = {args.variant: aggregated.get(args.variant)}
    else:
        selected_variants = {
            variant: variant_data
            for variant, variant_data in aggregated.items()
            if variant in VARIANT_LABELS
        }

    selected_variants = {variant: data for variant, data in selected_variants.items() if data}

    if not selected_variants:
        raise SystemExit("No benchmark rows were found for the selected variant.")

    print_summary(selected_variants)

    for variant, variant_data in selected_variants.items():
        variant_outdir = args.outdir / variant
        plot_variant(variant, variant_data, variant_outdir)
        for metric in PLOT_METRICS:
            print(f"Saved: {variant_outdir / f'{variant}_{metric}.png'}")

    if args.variant is None and len(selected_variants) > 1:
        plot_summary_comparison(selected_variants, args.outdir)
        for metric in METRICS:
            print(f"Saved: {args.outdir / 'summary' / f'summary_comparison_{metric}.png'}")

    print("Done.")


if __name__ == "__main__":
    main()