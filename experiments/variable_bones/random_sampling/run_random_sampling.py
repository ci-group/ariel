"""Randomly sample valid ARIEL morphologies with evolvable bone lengths.

No fitness, no selection, no crossover across generations.

Outputs:
    attempts.csv   - one row per attempted CPPN
    robots.csv     - one row per accepted morphology
    bones.csv      - one row per brick/bone
    graphs.jsonl   - optional full morphology graphs
    summary.json   - configuration and aggregate counts
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from random_sampling_common import (
    SamplingConfig,
    create_random_cppn,
    decode_graph,
    get_actuator_count,
    graph_statistics,
    graph_to_jsonable,
    is_physically_valid,
    iter_bones,
    make_id_manager,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample the ARIEL CPPN morphology space without selection."
    )
    parser.add_argument("--target-valid", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-modules", type=int, default=10)
    parser.add_argument("--initial-structural-mutations", type=int, default=3)
    parser.add_argument(
        "--require-actuated",
        action="store_true",
        help="Only accept morphologies with at least one actuator.",
    )
    parser.add_argument(
        "--require-brick",
        action="store_true",
        help="Only accept morphologies containing at least one brick/bone.",
    )
    parser.add_argument("--max-attempts", type=int, default=1_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("__data__") / "variable_bones" / "random_sampling",
    )
    parser.add_argument(
        "--save-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def open_writer(path: Path, fieldnames: list[str]):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return file, writer


def main() -> None:
    args = parse_args()
    if args.target_valid <= 0:
        raise ValueError("--target-valid must be > 0")

    args.output.mkdir(parents=True, exist_ok=True)

    config = SamplingConfig(
        seed=args.seed,
        max_modules=args.max_modules,
        initial_structural_mutations=args.initial_structural_mutations,
        require_actuated=args.require_actuated,
        require_brick=args.require_brick,
    )

    seed_everything(config.seed)
    id_manager = make_id_manager()

    attempts_file, attempts_writer = open_writer(
        args.output / "attempts.csv",
        [
            "attempt_id",
            "accepted",
            "status",
            "num_modules",
            "num_bricks",
            "num_hinges",
            "actuator_count",
        ],
    )

    robots_file, robots_writer = open_writer(
        args.output / "robots.csv",
        [
            "robot_id",
            "attempt_id",
            "seed",
            "num_modules",
            "num_bricks",
            "num_hinges",
            "actuator_count",
            "mean_bone_length_mm",
            "median_bone_length_mm",
            "min_bone_length_mm",
            "max_bone_length_mm",
            "std_bone_length_mm",
        ],
    )

    bones_file, bones_writer = open_writer(
        args.output / "bones.csv",
        [
            "robot_id",
            "attempt_id",
            "bone_index",
            "node_id",
            "length_mm",
        ],
    )

    graphs_file = None
    if args.save_graphs:
        graphs_file = (args.output / "graphs.jsonl").open("w", encoding="utf-8")

    accepted = 0
    attempts = 0
    status_counts = Counter()
    start = time.perf_counter()

    try:
        while accepted < args.target_valid and attempts < args.max_attempts:
            attempt_id = attempts
            attempts += 1

            stats = {
                "num_modules": "",
                "num_bricks": "",
                "num_hinges": "",
            }
            actuator_count = ""
            graph = None

            try:
                genome = create_random_cppn(
                    id_manager=id_manager,
                    initial_structural_mutations=config.initial_structural_mutations,
                )

                # Match the current body+brain experiment's feed-forward check.
                genome.get_node_ordering()

                graph = decode_graph(
                    genome=genome,
                    max_modules=config.max_modules,
                )

                if graph.number_of_nodes() == 0:
                    status = "empty_graph"
                else:
                    stats = graph_statistics(graph)

                    if not is_physically_valid(graph):
                        status = "physical_collision_or_compile_failure"
                    elif config.require_brick and stats["num_bricks"] == 0:
                        status = "no_bricks"
                    else:
                        actuator_count = get_actuator_count(graph)
                        if config.require_actuated and actuator_count <= 0:
                            status = "no_actuators"
                        else:
                            status = "accepted"

            except Exception as exc:
                status = "exception:" + type(exc).__name__

            is_accepted = status == "accepted"

            attempts_writer.writerow({
                "attempt_id": attempt_id,
                "accepted": int(is_accepted),
                "status": status,
                "num_modules": stats.get("num_modules", ""),
                "num_bricks": stats.get("num_bricks", ""),
                "num_hinges": stats.get("num_hinges", ""),
                "actuator_count": actuator_count,
            })
            status_counts[status] += 1

            if not is_accepted:
                continue

            robot_id = accepted
            accepted += 1

            robots_writer.writerow({
                "robot_id": robot_id,
                "attempt_id": attempt_id,
                "seed": config.seed,
                "num_modules": stats["num_modules"],
                "num_bricks": stats["num_bricks"],
                "num_hinges": stats["num_hinges"],
                "actuator_count": actuator_count,
                "mean_bone_length_mm": stats["mean_bone_length_mm"],
                "median_bone_length_mm": stats["median_bone_length_mm"],
                "min_bone_length_mm": stats["min_bone_length_mm"],
                "max_bone_length_mm": stats["max_bone_length_mm"],
                "std_bone_length_mm": stats["std_bone_length_mm"],
            })

            for bone in iter_bones(graph):
                bones_writer.writerow({
                    "robot_id": robot_id,
                    "attempt_id": attempt_id,
                    **bone,
                })

            if graphs_file is not None:
                graphs_file.write(
                    json.dumps({
                        "robot_id": robot_id,
                        "attempt_id": attempt_id,
                        "graph": graph_to_jsonable(graph),
                    })
                    + "\n"
                )

            if args.progress_every > 0 and accepted % args.progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = accepted / elapsed if elapsed > 0 else 0.0
                print(
                    f"Accepted {accepted:,}/{args.target_valid:,} "
                    f"after {attempts:,} attempts ({rate:.1f} valid robots/s)"
                )

    finally:
        attempts_file.close()
        robots_file.close()
        bones_file.close()
        if graphs_file is not None:
            graphs_file.close()

    elapsed = time.perf_counter() - start

    if accepted < args.target_valid:
        raise RuntimeError(
            f"Stopped at {accepted:,}/{args.target_valid:,} valid robots "
            f"after {attempts:,} attempts."
        )

    summary = {
        "experiment": "random_morphology_sampling",
        "seed": config.seed,
        "target_valid": args.target_valid,
        "accepted_valid": accepted,
        "total_attempts": attempts,
        "acceptance_rate": accepted / attempts if attempts else 0.0,
        "max_modules": config.max_modules,
        "initial_structural_mutations": config.initial_structural_mutations,
        "require_actuated": config.require_actuated,
        "require_brick": config.require_brick,
        "save_graphs": args.save_graphs,
        "elapsed_seconds": elapsed,
        "status_counts": dict(sorted(status_counts.items())),
    }

    write_json(args.output / "summary.json", summary)

    print()
    print(f"Finished: {accepted:,} valid robots from {attempts:,} attempts.")
    print(f"Acceptance rate: {summary['acceptance_rate']:.2%}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
