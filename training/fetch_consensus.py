import argparse
import json
from collections import Counter

import numpy as np
import wandb


def fetch_top_runs(sweep_id: str, metric: str, top_k: int = 5):
    """
    Fetch the top_k runs from the W&B sweep, sorted descending by `metric`.
    Returns a list of run.config dicts.
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    # fetch all runs in that sweep
    runs = list(sweep.runs)
    # sort by our metric (assumes larger is better)
    runs.sort(key=lambda r: r.summary.get(metric, float("-inf")), reverse=True)
    top_runs = runs[:top_k]
    print(f"Found {len(runs)} runs, picking top {len(top_runs)} by {metric}")
    return [run.config for run in top_runs]


def consensus(configs):
    """Build a consensus config from a list of flat hyperparam dicts."""
    keys = configs[0].keys()
    out = {}
    for k in keys:
        vals = [cfg[k] for cfg in configs if k in cfg]
        if not vals:
            continue
        # all ints?
        if all(isinstance(v, int) for v in vals):
            out[k] = Counter(vals).most_common(1)[0][0]
        # all floats?
        elif all(isinstance(v, (float, int)) for v in vals):
            out[k] = float(np.median(vals))
        # fallback categorical
        else:
            out[k] = Counter(vals).most_common(1)[0][0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
      "--sweep",
      required=True,
      help="W&B sweep ID, e.g. user/proj/sweep/4xyzabc1"
    )
    p.add_argument(
      "--metric",
      default="fall/f1",
      help="Which summary metric to sort on (must be in run.summary)."
    )
    p.add_argument(
      "--top_k",
      type=int,
      default=5,
      help="How many top runs to include in consensus."
    )
    args = p.parse_args()

    configs = fetch_top_runs(args.sweep, args.metric, args.top_k)
    cons = consensus(configs)
    with open("consensus_config.json", "w") as f:
        json.dump(cons, f, indent=2)
    print("Wrote consensus_config.json:")
    print(json.dumps(cons, indent=2))


if __name__ == "__main__":
    main()
