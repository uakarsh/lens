"""
Analyze per-class loss progression across epochs from training_log.json.

Usage:
  uv run python analyze_training_log.py --run-name=cifar_baseline_v3
  uv run python analyze_training_log.py --run-name=cifar_baseline_v3 --compare=cifar_augmented_v3
  uv run python analyze_training_log.py --run-name=cifar_baseline_v3 --class=cat
"""

import json
import os
import sys

RUNS_DIR = "runs"
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_log(run_name: str) -> list:
    path = os.path.join(RUNS_DIR, run_name, "training_log.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run train_cifar.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def print_class_loss_table(log: list, run_name: str):
    """Print per-class val loss for every epoch — spot plateaus and oscillations."""
    epochs = [e["epoch"] for e in log]
    print(f"\n{'='*80}")
    print(f"PER-CLASS VAL LOSS across epochs — {run_name}")
    print(f"{'='*80}")
    print(f"{'Class':>12} " + " ".join(f"ep{e:02d}" for e in epochs))
    print("-" * (14 + 6 * len(epochs)))

    # Track which epochs each class had a loss increase
    increases: dict[str, list[int]] = {}
    for cls in CIFAR10_CLASSES:
        losses = [e["per_class_val_loss"][cls] for e in log]
        up_epochs = [log[i+1]["epoch"] for i in range(len(losses)-1) if losses[i+1] > losses[i]]
        if up_epochs:
            increases[cls] = up_epochs

    for cls in CIFAR10_CLASSES:
        losses = [e["per_class_val_loss"][cls] for e in log]
        plateaued = len(losses) >= 3 and (max(losses[-3:]) - min(losses[-3:])) < 0.005
        oscillating = False
        if len(losses) >= 4:
            diffs = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
            signs = [1 if d > 0 else -1 for d in diffs[-4:]]
            oscillating = signs == [1,-1,1,-1] or signs == [-1,1,-1,1]

        flags = []
        if plateaued:   flags.append("⚠ PLATEAU")
        if oscillating: flags.append("~ OSCILLATE")
        if cls in increases:
            flags.append(f"↑ loss rose ep{increases[cls]}")

        # Bracket epochs where loss went up
        cells = []
        for i, l in enumerate(losses):
            if i > 0 and l > losses[i-1]:
                cells.append(f"[{l:.3f}]")
            else:
                cells.append(f" {l:.3f} ")
        row = " ".join(cells)
        flag_str = "  " + "  ".join(flags) if flags else ""
        print(f"  {cls:>10}  {row}{flag_str}")

    print()
    # Summary: ep1 vs last, drop%, # increases, diagnosis
    print(f"  {'Class':>10}  {'ep1':>7}  {'last':>7}  {'drop%':>6}  {'↑ count':>7}  diagnosis")
    print("  " + "-" * 72)
    for cls in CIFAR10_CLASSES:
        losses = [e["per_class_val_loss"][cls] for e in log]
        drop = losses[0] - losses[-1]
        pct  = drop / losses[0] * 100
        n_up = len(increases.get(cls, []))
        if pct < 5:
            diagnosis = "⚠  barely learning — label noise / boundary overlap"
        elif pct < 20 and n_up >= 2:
            diagnosis = "⚠  slow + unstable — contradictory gradients (annotation issue?)"
        elif pct < 20:
            diagnosis = "~  slow convergence — needs more data or epochs"
        elif n_up >= 2:
            diagnosis = "~  converging but unstable — boundary ambiguity"
        else:
            diagnosis = "✓  converging cleanly"
        print(f"  {cls:>10}  {losses[0]:>7.3f}  {losses[-1]:>7.3f}  {pct:>5.1f}%  {n_up:>7}  {diagnosis}")


def print_single_class(log: list, cls: str, run_name: str):
    """Deep dive into one class — loss + acc every epoch."""
    if cls not in CIFAR10_CLASSES:
        print(f"Unknown class '{cls}'. Choose from: {CIFAR10_CLASSES}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"CLASS DEEP DIVE: {cls} — {run_name}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>6}  {'Val Loss':>9}  {'Val Acc':>8}  {'Loss Δ':>8}  {'Acc Δ':>7}")
    print("  " + "-" * 45)

    prev_loss = None
    prev_acc  = None
    for e in log:
        loss = e["per_class_val_loss"][cls]
        acc  = e["per_class_val_acc"][cls]
        dl   = f"{loss - prev_loss:+.3f}" if prev_loss is not None else "    —"
        da   = f"{acc  - prev_acc:+.3f}"  if prev_acc  is not None else "   —"
        # Flag if loss went UP
        flag = " ← loss increased!" if (prev_loss is not None and loss > prev_loss) else ""
        print(f"  {e['epoch']:>6}  {loss:>9.3f}  {acc:>8.1%}  {dl:>8}  {da:>7}{flag}")
        prev_loss = loss
        prev_acc  = acc


def compare_runs(log1: list, log2: list, run1: str, run2: str):
    """Side-by-side final-epoch per-class loss for two runs."""
    print(f"\n{'='*70}")
    print(f"COMPARISON (final epoch): {run1}  vs  {run2}")
    print(f"{'='*70}")
    print(f"  {'Class':>10}  {run1:>14}  {run2:>14}  {'delta':>8}  {'winner':>12}")
    print("  " + "-" * 62)
    for cls in CIFAR10_CLASSES:
        l1 = log1[-1]["per_class_val_loss"][cls]
        l2 = log2[-1]["per_class_val_loss"][cls]
        d  = l2 - l1
        winner = run1 if l1 < l2 else run2
        print(f"  {cls:>10}  {l1:>14.3f}  {l2:>14.3f}  {d:>+8.3f}  {winner:>12}")


if __name__ == "__main__":
    args = sys.argv[1:]

    run_name = None
    compare  = None
    cls      = None
    for a in args:
        if a.startswith("--run-name="):  run_name = a.split("=",1)[1]
        if a.startswith("--compare="):   compare  = a.split("=",1)[1]
        if a.startswith("--class="):     cls      = a.split("=",1)[1]

    if not run_name:
        print("ERROR: --run-name=<name> required")
        print("Usage: uv run python analyze_training_log.py --run-name=cifar_baseline_v3")
        sys.exit(1)

    log = load_log(run_name)
    print_class_loss_table(log, run_name)

    if cls:
        print_single_class(log, cls, run_name)

    if compare:
        log2 = load_log(compare)
        compare_runs(log, log2, run_name, compare)
