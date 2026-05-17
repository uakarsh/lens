"""
CIFAR-10 experiment script — mirrors train_and_analyze.py structure.

Runs:
  baseline (no augmentation):
    uv run python train_cifar.py --run-name=cifar_baseline --no-augment

  augmented:
    uv run python train_cifar.py --run-name=cifar_augmented

Each run stores results in runs/<run-name>/:
  model.pth, failures.json, predictions.json, analysis.json

Cross-run comparison is appended to runs/compare.json (same ledger as MNIST).
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Vehicles: benefit from geometric augmentation (flip, crop)
# Animals: fine-grained texture — only mild flip, no crop/color distortion
VEHICLE_CLASSES = {0, 1, 8, 9}   # airplane, automobile, ship, truck
ANIMAL_CLASSES  = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse

RUNS_DIR = "runs"
COMPARE_PATH = "runs/compare.json"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ClassConditionalCIFAR(torch.utils.data.Dataset):
    """Wraps CIFAR-10 and applies different augmentations per class group."""

    def __init__(self, root, train, download, to_tensor_normalize):
        from torchvision.transforms import functional as TF
        import random
        self._tf = TF
        self._random = random

        # Load raw PIL images — no transform at dataset level
        self.ds = datasets.CIFAR10(root, train=train, download=download, transform=None)
        self.to_tensor_norm = to_tensor_normalize

        self.vehicle_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
        self.animal_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),  # subtle — fewer animals are horizontally symmetric
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if label in VEHICLE_CLASSES:
            img = self.vehicle_aug(img)
        else:
            img = self.animal_aug(img)
        img = self.to_tensor_norm(img)
        return img, label


def run_paths(run_name: str) -> dict:
    base = os.path.join(RUNS_DIR, run_name)
    os.makedirs(base, exist_ok=True)
    return {
        "dir":         base,
        "model":       os.path.join(base, "model.pth"),
        "failures":    os.path.join(base, "failures.json"),
        "predictions": os.path.join(base, "predictions.json"),
        "analysis":    os.path.join(base, "analysis.json"),
    }


# ─── Model ────────────────────────────────────────────────────────────────────
class CifarCNN(nn.Module):
    """Simple CNN for CIFAR-10: 3 conv blocks + 2 FC layers."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─── Step 1: Train ────────────────────────────────────────────────────────────
def train_model(paths: dict, augment: str = "class_conditional", epochs: int = 15):
    """
    augment: "none" | "full" | "class_conditional"
      none              — no augmentation (baseline)
      full              — same augmentation for all classes
      class_conditional — vehicles get full aug, animals get flip-only
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    val_transform = to_tensor_norm

    if augment == "class_conditional":
        train_ds = ClassConditionalCIFAR(".", train=True, download=True, to_tensor_normalize=to_tensor_norm)
    elif augment == "full":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
        train_ds = datasets.CIFAR10(".", train=True, download=True, transform=train_transform)
    else:  # none
        train_ds = datasets.CIFAR10(".", train=True, download=True, transform=to_tensor_norm)

    test_ds = datasets.CIFAR10(".", train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = CifarCNN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mean = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction="none")  # per-sample loss for class breakdown

    print("=" * 60)
    print(f"STEP 1 — Training CNN on CIFAR-10 (augment={augment!r}, epochs={epochs}, device={DEVICE})")
    print("=" * 60)

    epoch_log = []  # [{epoch, total_loss, per_class_loss: {class: avg_loss}, per_class_acc: {class: acc}}]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion_mean(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Per-class loss + accuracy on val set
        model.eval()
        class_loss_sum = [0.0] * 10
        class_correct  = [0] * 10
        class_total    = [0] * 10
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                losses = criterion_none(logits, labels)  # (B,)
                preds  = logits.argmax(dim=1)
                for j in range(len(labels)):
                    c = int(labels[j])
                    class_loss_sum[c] += losses[j].item()
                    class_total[c]    += 1
                    if int(preds[j]) == c:
                        class_correct[c] += 1
        model.train()

        per_class_loss = {CIFAR10_CLASSES[i]: round(class_loss_sum[i] / class_total[i], 4) for i in range(10)}
        per_class_acc_ep = {CIFAR10_CLASSES[i]: round(class_correct[i] / class_total[i], 4) for i in range(10)}

        # Print compact per-class loss row
        loss_str = "  ".join(f"{CIFAR10_CLASSES[i][0:4]}={per_class_loss[CIFAR10_CLASSES[i]]:.3f}" for i in range(10))
        print(f"  Epoch {epoch}/{epochs} — loss: {avg_loss:.4f} | {loss_str}")

        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "per_class_val_loss": per_class_loss,
            "per_class_val_acc":  per_class_acc_ep,
        })

    # Save training log
    log_path = os.path.join(paths["dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(epoch_log, f, indent=2)
    print(f"\nTraining log saved to {log_path}")

    # Final per-class accuracy from last epoch
    per_class_acc = epoch_log[-1]["per_class_val_acc"]
    print("\nFinal per-class accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  {cls:>12}: {acc:.1%}")

    torch.save(model.state_dict(), paths["model"])
    print(f"\nModel saved to {paths['model']}")
    return model, per_class_acc


# ─── Step 2: Collect failures + all predictions ───────────────────────────────
def collect_failures(model, paths: dict):
    print("\n" + "=" * 60)
    print("STEP 2 — Collecting failures")
    print("=" * 60)

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_ds     = datasets.CIFAR10(".", train=False, download=False, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model.eval()
    failures = []
    all_predictions = []
    idx_offset = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = logits.argmax(dim=1).cpu()
            for i in range(len(labels)):
                true      = int(labels[i])
                pred      = int(preds[i])
                conf_pred = float(probs[i][pred])
                conf_true = float(probs[i][true])
                all_predictions.append({
                    "image_idx":            idx_offset + i,
                    "true_label":           true,
                    "true_class":           CIFAR10_CLASSES[true],
                    "predicted_label":      pred,
                    "predicted_class":      CIFAR10_CLASSES[pred],
                    "correct":              true == pred,
                    "confidence_predicted": conf_pred,
                    "confidence_true":      conf_true,
                })
                if true != pred:
                    failures.append({
                        "image_idx":            idx_offset + i,
                        "true_label":           true,
                        "true_class":           CIFAR10_CLASSES[true],
                        "predicted_label":      pred,
                        "predicted_class":      CIFAR10_CLASSES[pred],
                        "confidence_predicted": conf_pred,
                        "confidence_true":      conf_true,
                    })
            idx_offset += len(labels)

    conf_pairs = Counter((f["true_class"], f["predicted_class"]) for f in failures)
    print(f"\nTotal failures: {len(failures)} / 10,000")
    print("\nTop confused pairs:")
    for (t, p), cnt in conf_pairs.most_common(10):
        print(f"  {t:>12} → {p}: {cnt}")

    with open(paths["failures"], "w") as f:
        json.dump(failures, f)
    with open(paths["predictions"], "w") as f:
        json.dump(all_predictions, f)
    print(f"\nSaved failures to    {paths['failures']}")
    print(f"Saved predictions to {paths['predictions']}")
    return failures


# ─── Step 3: Metrics + Save ───────────────────────────────────────────────────
def compute_metrics(predictions: list) -> dict:
    tp = [0]*10; fp = [0]*10; fn = [0]*10; total = [0]*10
    for p in predictions:
        t, pred = p["true_label"], p["predicted_label"]
        total[t] += 1
        if t == pred:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[pred] += 1

    metrics = {}
    print(f"\n{'Class':>12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 52)
    for i in range(10):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics[CIFAR10_CLASSES[i]] = {"precision": prec, "recall": rec, "f1": f1, "support": total[i]}
        print(f"  {CIFAR10_CLASSES[i]:>10} {prec:>10.1%} {rec:>8.1%} {f1:>8.1%} {total[i]:>9}")
    return metrics


def save_analysis(failures, predictions, per_class_acc, metrics, paths: dict, experiment_id: str):
    conf_pairs = Counter((f["true_class"], f["predicted_class"]) for f in failures)
    analysis = {
        "experiment_id": experiment_id,
        "dataset": "cifar10",
        "summary": {
            "total_test":     10000,
            "total_failures": len(failures),
            "failure_rate":   round(len(failures) / 10000, 4),
            "per_class_accuracy": per_class_acc,
        },
        "confused_pairs": [
            {"true": t, "predicted": p, "count": cnt}
            for (t, p), cnt in conf_pairs.most_common(20)
        ],
        "metrics": metrics,
        "failures": failures,
    }
    with open(paths["analysis"], "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {paths['analysis']}")

    # Update shared compare ledger
    os.makedirs(RUNS_DIR, exist_ok=True)
    compare = {}
    if os.path.exists(COMPARE_PATH):
        with open(COMPARE_PATH) as f:
            compare = json.load(f)
    compare[experiment_id] = {
        "experiment_id":  experiment_id,
        "dataset":        "cifar10",
        "total_failures": len(failures),
        "failure_rate":   analysis["summary"]["failure_rate"],
        "per_class_accuracy": per_class_acc,
        "top_confused_pairs": analysis["confused_pairs"][:5],
        "run_dir":        paths["dir"],
    }
    with open(COMPARE_PATH, "w") as f:
        json.dump(compare, f, indent=2)
    print(f"Comparison ledger updated at {COMPARE_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(experiment_id: str, augment: str = "class_conditional", epochs: int = 15):
    paths = run_paths(experiment_id)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_id}  (dir: {paths['dir']})")
    print(f"{'='*60}")

    # Step 1 — Train or load
    if os.path.exists(paths["model"]):
        print(f"Found existing model at {paths['model']}, loading...")
        model = CifarCNN().to(DEVICE)
        model.load_state_dict(torch.load(paths["model"], map_location=DEVICE))
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std  = (0.2470, 0.2435, 0.2616)
        val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])
        test_ds = datasets.CIFAR10(".", train=False, download=True, transform=val_transform)
        loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
        class_correct = [0]*10; class_total = [0]*10
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu()
                for label, pred in zip(labels, preds):
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        per_class_acc = {CIFAR10_CLASSES[i]: class_correct[i]/class_total[i] for i in range(10)}
    else:
        model, per_class_acc = train_model(paths, augment=augment, epochs=epochs)

    # Step 2 — Collect failures
    if os.path.exists(paths["failures"]) and os.path.exists(paths["predictions"]):
        print(f"\nFound cached failures/predictions, loading...")
        with open(paths["failures"])    as f: failures    = json.load(f)
        with open(paths["predictions"]) as f: predictions = json.load(f)
        print(f"Loaded {len(failures)} failures, {len(predictions)} predictions")
    else:
        failures = collect_failures(model, paths)
        with open(paths["predictions"]) as f:
            predictions = json.load(f)

    # Step 3 — Metrics
    print("\n" + "=" * 60)
    print("STEP 3 — Per-class metrics")
    print("=" * 60)
    metrics = compute_metrics(predictions)

    # Step 4 — Save
    save_analysis(failures, predictions, per_class_acc, metrics, paths, experiment_id)


if __name__ == "__main__":
    args = sys.argv[1:]

    epochs = 15
    for a in args:
        if a.startswith("--epochs="):
            epochs = int(a.split("=", 1)[1])

    # --augment=none | full | class_conditional (default)
    augment = "class_conditional"
    for a in args:
        if a.startswith("--augment="):
            augment = a.split("=", 1)[1]
    if "--no-augment" in args:
        augment = "none"

    run_name = None
    for a in args:
        if a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
    if run_name is None:
        print("ERROR: --run-name=<experiment_id> is required")
        print("Examples:")
        print("  uv run python train_cifar.py --run-name=cifar_baseline --no-augment")
        print("  uv run python train_cifar.py --run-name=cifar_full_aug --augment=full")
        print("  uv run python train_cifar.py --run-name=cifar_class_cond  # default")
        sys.exit(1)

    main(experiment_id=run_name, augment=augment, epochs=epochs)
