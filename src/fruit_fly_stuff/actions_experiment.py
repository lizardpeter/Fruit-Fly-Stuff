"""Cloud-friendly synthetic neural-memory experiment; no local computer required.

Designed to be invoked by GitHub Actions. The current circuit is three synthetic
neurons, NOT a biologically validated fruit-fly network.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .demo import cue_response, make_toy_brain, train


def run_experiment(checkpoint: Path, *, trials: int = 3, reset: bool = False) -> dict:
    """Continue learning from checkpoint, or reset if explicitly requested."""
    if not 1 <= trials <= 12:
        raise ValueError("trials must be between 1 and 12")
    had_checkpoint = checkpoint.is_file()
    restored_previous = had_checkpoint and not reset
    brain = load_checkpoint(checkpoint) if restored_previous else make_toy_brain()

    # Cue testing resets activity/eligibility; do it on copies to preserve
    # the actual persisted neural state and unsaved eligibility traces.
    before_response = cue_response(deepcopy(brain))
    weight_before = float(brain.weights[0])
    steps_before = brain.steps

    train(brain, trials=trials)

    weight_after = float(brain.weights[0])
    after_response = cue_response(deepcopy(brain))
    save_checkpoint(brain, checkpoint)
    reloaded = load_checkpoint(checkpoint)
    restored_response = cue_response(deepcopy(reloaded))

    if restored_response != after_response:
        raise AssertionError("checkpoint restore changed cue recall")

    return {
        "model": "synthetic three-neuron circuit, not a fruit-fly connectome",
        "loaded_previous_memory": bool(restored_previous),
        "reset_requested": bool(reset),
        "trials_this_run": trials,
        "steps_before": steps_before,
        "steps_after": brain.steps,
        "cue_weight_before": round(weight_before, 6),
        "cue_weight_after": round(weight_after, 6),
        "cue_response_before": bool(before_response),
        "cue_response_after_restore": bool(restored_response),
        "checkpoint": str(checkpoint),
    }


def render_summary(result: dict) -> str:
    fields = (
        ("Previous saved memory restored", result["loaded_previous_memory"]),
        ("Reset requested", result["reset_requested"]),
        ("Training trials this run", result["trials_this_run"]),
        ("Total simulation steps", result["steps_after"]),
        ("Cue synapse weight before", result["cue_weight_before"]),
        ("Cue synapse weight after", result["cue_weight_after"]),
        ("Cue caused output before training", result["cue_response_before"]),
        ("Cue caused output after save and restore", result["cue_response_after_restore"]),
    )
    rows = "\n".join(
        f"| {label} | {str(value).lower() if isinstance(value, bool) else value} |"
        for label, value in fields
    )
    return (
        "## Synthetic neural-memory experiment\n\n"
        "> This tests simulator persistence and conditioning, **not biological fly behavior**.\n\n"
        "| Measurement | Result |\n|---|---|\n" + rows + "\n\n"
        "State is stored as a small checkpoint on the simulation-state branch, "
        "and a copy is attached as an Actions artifact.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("state/toy-brain.npz"))
    parser.add_argument("--trials", type=int, choices=(1, 3, 6, 12), default=3)
    parser.add_argument("--reset", choices=("true", "false"), default="false")
    parser.add_argument("--report", type=Path, default=Path("results/experiment.json"))
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    report = run_experiment(args.checkpoint, trials=args.trials, reset=args.reset == "true")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as file:
            file.write(render_summary(report) + "\n")


if __name__ == "__main__":
    main()
