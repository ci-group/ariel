"""Re-tune fragile hover priors in the hex library at a higher CMA budget.

~20% of the v1 library morphs have `median_score < 400` even though their
best-seed `score` passes the Stage-1 gate — high-variance CMA solutions
whose priors are fragile under residual perturbation (the most likely
source of the "residual learns to fight the prior" failure mode). This
script re-runs the 35c/36 CMA-ES on exactly those morphs with a larger
budget and updates `library.npz` **in place**, keeping a timestamped
backup next to it. A new result is only accepted if it improves
`median_score` without regressing `score` by more than 10%.

Run:
    uv run examples/spear/library/40_retune_fragile_priors.py \\
        --budget 1200 --population 128

Pass `--budget 100 --limit 1` for a fast smoke test.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from examples.spear.under_development.library.hex_sampler import sample_feasible  # noqa: E402

_lib36 = importlib.import_module("36_build_hover_library")


def main() -> int:
    p = argparse.ArgumentParser(description="Re-tune fragile hover priors")
    p.add_argument("--library", default="__data__/hex_library/v1/library.npz")
    p.add_argument("--threshold", type=float, default=400.0,
                   help="Re-tune morphs with median_score below this.")
    p.add_argument("--budget", type=int, default=1200,
                   help="CMA generations per morph (3x the v1 build's 400).")
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--sigma0", type=float, default=0.4)
    p.add_argument("--zero-init", action="store_true")
    p.add_argument("--action-scale", type=float, default=0.4)
    p.add_argument("--episode-steps", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=43,
                   help="Base CMA seed. Deliberately != the v1 build's 42 "
                        "so the re-run explores a different sample path.")
    p.add_argument("--limit", type=int, default=0,
                   help="Re-tune at most this many morphs (0 = all fragile).")
    args = p.parse_args()

    lib_path = Path(args.library)
    d = dict(np.load(lib_path))
    fragile = np.where(d["median_score"] < args.threshold)[0]
    if args.limit > 0:
        fragile = fragile[: args.limit]
    if len(fragile) == 0:
        print(f"no morphs below median_score {args.threshold}; nothing to do")
        return 0
    print(f"{len(fragile)} fragile morphs "
          f"(median_score < {args.threshold}): indices {fragile.tolist()}")

    backup = lib_path.with_name(
        f"library_pre_retune_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    )
    shutil.copy2(lib_path, backup)
    print(f"backed up library → {backup}")

    # Round-trip the morph objects the same way 36/37 do.
    sampled = sample_feasible(200, seed=42, stratify=True)
    by_seed = {m.seed: m for m in sampled}

    accepted, rejected = [], []
    t_all = time.time()
    for n, i in enumerate(fragile):
        seed = int(d["morph_seed"][i])
        morph = by_seed.get(seed)
        if morph is None:
            print(f"  [{i}] seed {seed} not recoverable from sampler — skipped")
            continue
        old_score = float(d["score"][i])
        old_median = float(d["median_score"][i])
        t0 = time.time()
        result = _lib36.train_one_morph(
            morph,
            budget=args.budget, population=args.population,
            max_steps=args.episode_steps,
            seed=args.seed * 1_000_003 + seed, device=args.device,
            sigma0=args.sigma0, zero_init=args.zero_init,
            action_scale=args.action_scale,
        )
        better_median = result.median_score > old_median
        score_ok = result.score >= 0.9 * old_score
        tag = "ACCEPT" if (better_median and score_ok) else "reject"
        print(f"  [{n + 1}/{len(fragile)}] idx={i} seed={seed}  "
              f"median {old_median:.1f} → {result.median_score:.1f}  "
              f"score {old_score:.1f} → {result.score:.1f}  "
              f"gens={result.generations_run}  "
              f"({time.time() - t0:.1f}s)  {tag}")
        if better_median and score_ok:
            d["cmaes_params"][i] = result.cmaes_params.astype(np.float32)
            d["score"][i] = np.float32(result.score)
            d["median_score"][i] = np.float32(result.median_score)
            d["generations_run"][i] = np.int32(result.generations_run)
            d["train_time_s"][i] = np.float32(result.train_time_s)
            accepted.append(int(i))
        else:
            rejected.append(int(i))

    # Atomic in-place update (same tmp+rename dance as 36).
    tmp = lib_path.with_name("library.tmp")
    np.savez(str(tmp), **d)
    tmp.with_name(tmp.name + ".npz").rename(lib_path)

    manifest_path = lib_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("retunes", []).append({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": args.threshold,
            "budget": args.budget,
            "population": args.population,
            "seed": args.seed,
            "accepted_indices": accepted,
            "rejected_indices": rejected,
            "backup": backup.name,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))

    ms = d["median_score"]
    print(f"\naccepted {len(accepted)}/{len(fragile)} re-tunes "
          f"in {(time.time() - t_all) / 60:.1f} min")
    print(f"median_score p10 now {np.percentile(ms, 10):.1f} "
          f"(was 298.2); {int((ms < args.threshold).sum())} morphs "
          f"still below {args.threshold}")
    print(f"updated {lib_path} in place (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
