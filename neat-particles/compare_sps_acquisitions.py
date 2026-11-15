from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from genome_targets import load_genome_target
from sps_selection import load_particle_config, run_auto_sps_selection


ACQUISITIONS = ("gp", "surrogate")
CSV_FIELDS = [
    "mode",
    "seed",
    "stop_reason",
    "steps",
    "elapsed_seconds",
    "final_distance",
    "final_histogram_distance",
    "final_ssim_distance",
    "final_design_space",
    "switch_step",
    "switch_reason",
    "initial_best_distance",
]


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config-generic.ini")


def _default_output_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-runs", "acquisition-comparison")


def paired_seeds(seed_start: int, runs: int) -> List[int]:
    """Return the paired seeds shared by both acquisition modes."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    return list(range(seed_start, seed_start + runs))


def result_to_row(mode: str, seed: int, result) -> Dict[str, object]:
    """Convert one auto SPS result into the flat comparison row format."""
    return {
        "mode": mode,
        "seed": seed,
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "elapsed_seconds": result.elapsed_seconds,
        "final_distance": result.final_distance,
        "final_histogram_distance": result.final_histogram_distance,
        "final_ssim_distance": result.final_ssim_distance,
        "final_design_space": result.final_design_space,
        "switch_step": result.switch_step,
        "switch_reason": result.switch_reason,
        "initial_best_distance": result.initial_best_distance,
    }


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """Compute aggregate statistics by acquisition mode."""
    summary: Dict[str, Dict[str, object]] = {}
    for mode in ACQUISITIONS:
        mode_rows = [row for row in rows if row["mode"] == mode]
        stop_reason_counts: Dict[str, int] = {}
        for row in mode_rows:
            reason = str(row["stop_reason"])
            stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1

        summary[mode] = {
            "runs": len(mode_rows),
            "stop_reason_counts": stop_reason_counts,
            "avg_steps": _average(float(row["steps"]) for row in mode_rows),
            "avg_elapsed_seconds": _average(float(row["elapsed_seconds"]) for row in mode_rows),
            "avg_final_distance": _average(float(row["final_distance"]) for row in mode_rows),
            "best_final_distance": min((float(row["final_distance"]) for row in mode_rows), default=None),
            "worst_final_distance": max((float(row["final_distance"]) for row in mode_rows), default=None),
        }
    return summary


def paired_comparisons(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Compare GP and surrogate outcomes seed by seed."""
    by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["mode"])] = row

    comparisons = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if not all(mode in pair for mode in ACQUISITIONS):
            continue
        gp_distance = float(pair["gp"]["final_distance"])
        surrogate_distance = float(pair["surrogate"]["final_distance"])
        if gp_distance < surrogate_distance:
            winner = "gp"
        elif surrogate_distance < gp_distance:
            winner = "surrogate"
        else:
            winner = "tie"
        comparisons.append({
            "seed": seed,
            "winner": winner,
            "gp_final_distance": gp_distance,
            "surrogate_final_distance": surrogate_distance,
            "final_distance_delta_gp_minus_surrogate": gp_distance - surrogate_distance,
            "gp_steps": pair["gp"]["steps"],
            "surrogate_steps": pair["surrogate"]["steps"],
        })
    return comparisons


def run_comparison(target, config, seeds: Sequence[int], threshold: float, max_steps: int) -> List[Dict[str, object]]:
    """Run both acquisition modes for each paired seed."""
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for mode in ACQUISITIONS:
            result = run_auto_sps_selection(
                target,
                config,
                seed=seed,
                threshold=threshold,
                max_steps=max_steps,
                acquisition=mode,
            )
            rows.append(result_to_row(mode, seed, result))
            print(
                f"{mode} seed={seed} stop={result.stop_reason} steps={result.steps} "
                f"distance={result.final_distance:.6f} elapsed={result.elapsed_seconds:.3f}s"
            )
    return rows


def write_outputs(args, rows: Sequence[Dict[str, object]]) -> Tuple[str, str]:
    """Write aggregate JSON and flat CSV outputs."""
    output_dir = os.path.abspath(args.output_dir or _default_output_dir())
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"acquisition_comparison_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"acquisition_comparison_{timestamp}.csv")

    summary = summarize_rows(rows)
    pairs = paired_comparisons(rows)
    payload = {
        "target_path": os.path.abspath(args.target),
        "config_path": os.path.abspath(args.config),
        "runs_per_mode": args.runs,
        "threshold": args.threshold,
        "max_steps": args.max_steps,
        "seed_start": args.seed_start,
        "acquisitions": list(ACQUISITIONS),
        "summary": summary,
        "paired_comparisons": pairs,
        "rows": list(rows),
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(json_path), os.path.abspath(csv_path)


def _average(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GP and surrogate SPS acquisition over paired auto runs.")
    parser.add_argument("--target", required=True, help="Path to an exported target genome JSON file.")
    parser.add_argument("--config", default=_default_config_path(), help="Path to particle NEAT config.")
    parser.add_argument("--runs", type=int, default=50, help="Number of paired seeds to run per acquisition mode.")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_particle_config(args.config)
    target, _target_data = load_genome_target(args.target, config)
    seeds = paired_seeds(args.seed_start, args.runs)
    rows = run_comparison(target, config, seeds, args.threshold, args.max_steps)
    json_path, csv_path = write_outputs(args, rows)

    print(f"Aggregate JSON: {json_path}")
    print(f"CSV results: {csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
