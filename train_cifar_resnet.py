"""
CIFAR-10 error analysis pipeline using pretrained ResNet-18.

Mirrors train_and_analyze.py structure exactly — same runs/ layout,
same compare.json ledger, same LLM hypothesis + clustering steps.

Steps:
  1. Fine-tune ResNet-18 (pretrained on ImageNet) on CIFAR-10
  2. Collect failures + all predictions
  3. LLM hypotheses (CIFAR-aware prompt, cached per image_idx)
  4. Cluster hypotheses with BGE embeddings
  5. Label clusters
  6. Save analysis + update compare ledger

Runs:
  # Baseline — no augmentation, 3 epochs fine-tune
  uv run python train_cifar_resnet.py --run-name=resnet_baseline --no-augment --epochs=3

  # Augmented
  uv run python train_cifar_resnet.py --run-name=resnet_augmented --epochs=3

  # Skip LLM (just train + collect failures + metrics)
  uv run python train_cifar_resnet.py --run-name=resnet_baseline --no-augment --no-llm

  # Test mode (2 failures, skip clustering)
  uv run python train_cifar_resnet.py --run-name=resnet_test --epochs=1 --test
"""

import base64
import io
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

import ollama
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

OLLAMA_MODEL = "qwen3-vl:8b"
RUNS_DIR = "runs"
COMPARE_PATH = "runs/compare.json"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Raw test images for review UI (loaded once)
_test_images: dict[int, np.ndarray] = {}


def run_paths(run_name: str) -> dict:
    base = os.path.join(RUNS_DIR, run_name)
    os.makedirs(base, exist_ok=True)
    return {
        "dir":     base,
        "model":   os.path.join(base, "model.pth"),
        "failures":    os.path.join(base, "failures.json"),
        "predictions": os.path.join(base, "predictions.json"),
        "llm_cache":   os.path.join(base, "llm_cache.json"),
        "hyp":         os.path.join(base, "failures_with_hypotheses.json"),
        "analysis":    os.path.join(base, "analysis.json"),
        "training_log":os.path.join(base, "training_log.json"),
    }


# ─── Step 1: Fine-tune ResNet-18 ──────────────────────────────────────────────
def train_model(paths: dict, augment: bool = True, epochs: int = 3):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_ds    = datasets.CIFAR10(".", train=True,  download=True, transform=train_transform)
    test_ds     = datasets.CIFAR10(".", train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    # ResNet-18 pretrained — replace final FC for 10 classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    # Fine-tune: higher LR for FC, lower for backbone
    optimizer = optim.AdamW([
        {"params": model.fc.parameters(),  "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mean = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    print("=" * 60)
    print(f"STEP 1 — Fine-tuning ResNet-18 on CIFAR-10 (augment={augment}, epochs={epochs}, device={DEVICE})")
    print("=" * 60)

    epoch_log = []

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

        # Per-class val loss + accuracy
        model.eval()
        class_loss_sum = [0.0] * 10
        class_correct  = [0] * 10
        class_total    = [0] * 10
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                losses = criterion_none(logits, labels)
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
        loss_str = "  ".join(f"{CIFAR10_CLASSES[i][:4]}={per_class_loss[CIFAR10_CLASSES[i]]:.3f}" for i in range(10))
        print(f"  Epoch {epoch}/{epochs} — loss: {avg_loss:.4f} | {loss_str}")

        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "per_class_val_loss": per_class_loss,
            "per_class_val_acc":  per_class_acc_ep,
        })

    with open(paths["training_log"], "w") as f:
        json.dump(epoch_log, f, indent=2)
    print(f"\nTraining log saved to {paths['training_log']}")

    per_class_acc = epoch_log[-1]["per_class_val_acc"]
    print("\nFinal per-class accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  {cls:>12}: {acc:.1%}")

    torch.save(model.state_dict(), paths["model"])
    print(f"\nModel saved to {paths['model']}")
    return model, per_class_acc


# ─── Step 2: Collect failures ─────────────────────────────────────────────────
def collect_failures(model, paths: dict):
    print("\n" + "=" * 60)
    print("STEP 2 — Collecting failures")
    print("=" * 60)

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])
    test_ds     = datasets.CIFAR10(".", train=False, download=False, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Also cache raw PIL images for review UI
    raw_ds = datasets.CIFAR10(".", train=False, download=False, transform=None)
    for idx in range(len(raw_ds)):
        img, _ = raw_ds[idx]
        _test_images[idx] = np.array(img)  # (32, 32, 3) uint8

    model.eval()
    failures = []
    all_predictions = []
    idx_offset = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = logits.argmax(dim=1).cpu()
            for i in range(len(labels)):
                true      = int(labels[i])
                pred      = int(preds[i])
                conf_pred = float(probs[i][pred])
                conf_true = float(probs[i][true])
                # Top-3 predictions for richer LLM context
                top3 = torch.topk(probs[i], 3)
                top3_preds = [
                    {"class": CIFAR10_CLASSES[int(top3.indices[k])], "confidence": round(float(top3.values[k]), 3)}
                    for k in range(3)
                ]
                entry = {
                    "image_idx":            idx_offset + i,
                    "true_label":           true,
                    "true_class":           CIFAR10_CLASSES[true],
                    "predicted_label":      pred,
                    "predicted_class":      CIFAR10_CLASSES[pred],
                    "correct":              true == pred,
                    "confidence_predicted": round(conf_pred, 4),
                    "confidence_true":      round(conf_true, 4),
                    "top3":                 top3_preds,
                }
                all_predictions.append(entry)
                if true != pred:
                    failures.append({k: v for k, v in entry.items() if k != "correct"})
            idx_offset += len(labels)

    conf_pairs = Counter((f["true_class"], f["predicted_class"]) for f in failures)
    print(f"\nTotal failures: {len(failures)} / 10,000  ({len(failures)/10000:.2%} error rate)")
    print("\nTop confused pairs:")
    for (t, p), cnt in conf_pairs.most_common(10):
        print(f"  {t:>12} → {p}: {cnt}")

    # Per-class P/R/F1
    tp=[0]*10; fp=[0]*10; fn=[0]*10; total=[0]*10
    for p in all_predictions:
        t, pred = p["true_label"], p["predicted_label"]
        total[t] += 1
        if t == pred: tp[t] += 1
        else: fn[t] += 1; fp[pred] += 1

    print(f"\n{'Class':>12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 42)
    for i in range(10):
        prec = tp[i]/(tp[i]+fp[i]) if (tp[i]+fp[i]) > 0 else 0
        rec  = tp[i]/(tp[i]+fn[i]) if (tp[i]+fn[i]) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        print(f"  {CIFAR10_CLASSES[i]:>10} {prec:>10.1%} {rec:>8.1%} {f1:>8.1%}")

    with open(paths["failures"], "w") as f:
        json.dump(failures, f, indent=2)
    with open(paths["predictions"], "w") as f:
        json.dump(all_predictions, f)
    print(f"\nSaved {len(failures)} failures → {paths['failures']}")
    return failures


# ─── Step 3: LLM Hypotheses ───────────────────────────────────────────────────
def _img_to_b64(arr: np.ndarray) -> str:
    """Upscale 32x32 RGB array to 128x128 PNG, return base64."""
    pil = Image.fromarray(arr).resize((128, 128), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3-vl emits even when think=False."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def generate_hypothesis(failure: dict) -> str:
    true_cls  = failure["true_class"]
    pred_cls  = failure["predicted_class"]
    conf_pred = failure["confidence_predicted"]
    conf_true = failure["confidence_true"]
    top3      = failure.get("top3", [])
    conf_gap  = conf_pred - conf_true

    top3_str = ", ".join(f"{t['class']} ({t['confidence']:.1%})" for t in top3)

    if conf_gap > 0.5:
        certainty = "very confidently wrong — the model has almost no doubt"
    elif conf_gap > 0.2:
        certainty = "moderately confident in the wrong answer"
    else:
        certainty = "uncertain — this was a near-tie between classes"

    prompt = f"""You are reviewing a failure from an image classifier trained on CIFAR-10. Your job is to diagnose WHY the model got this wrong — the same way a careful human annotator would.

True class:      {true_cls}
Predicted class: {pred_cls}
Confidence in wrong prediction ({pred_cls}): {conf_pred:.1%}
Confidence in correct class ({true_cls}):    {conf_true:.1%}
Top-3 predictions: {top3_str}
Assessment: The model was {certainty}.

Look at the image carefully, then answer these three questions in order:

1. DATA ISSUE? Does this look like a labeling mistake or an ambiguous image that a human could also get wrong? (e.g. the image genuinely looks more like {pred_cls} than {true_cls}, or is cropped/corrupted)

2. VISUAL CAUSE: What specific visual property in this image made it look like {pred_cls}? Name the exact thing you see — background color, object pose, partial occlusion, crop that cut off a key feature, texture similarity. Be concrete, not generic.

3. FIX: Given the visual cause, what would help the model learn to distinguish these? Think in terms of: more training examples of this edge case, a specific augmentation (rotation, zoom, color jitter, cropping), or removing this sample if it's a labeling error.

Be specific and preserve detail — especially for the visual cause. Do not say "I" or mention being an AI."""

    # Build message with image attached
    img_arr = _test_images.get(failure["image_idx"])
    user_msg = {
        "role": "user",
        "content": prompt,
    }
    if img_arr is not None:
        user_msg["images"] = [_img_to_b64(img_arr)]

    # qwen3-vl:8b has a broken thinking-toggle template (ollama issue #14798).
    # Workaround: pre-fill the assistant turn with an empty <think> block so the
    # model skips internal reasoning and writes directly to content.
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at diagnosing image classifier failures. "
                    "You think like a human annotator: you look at the image, identify the specific visual property that caused confusion, "
                    "then suggest a concrete fix — whether that's a data fix (remove/relabel), an augmentation (rotation, zoom, crop, color), "
                    "or a note that the model needs more examples of this edge case. "
                    "Never give vague answers like 'similar features' — always name what you actually see."
                ),
            },
            user_msg,
            {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
        ],
        options={"temperature": 0.1, "num_predict": 300},
        stream=False,
    )
    return _strip_think(response.message.content or "")


def generate_hypotheses(failures: list, paths: dict) -> list:
    print("\n" + "=" * 60)
    print("STEP 3 — Generating LLM hypotheses")
    print("=" * 60)

    cache = {}
    if os.path.exists(paths["llm_cache"]):
        with open(paths["llm_cache"]) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached hypotheses")

    failures_with_hyp = []
    for i, failure in enumerate(failures):
        key = str(failure["image_idx"])
        if key in cache and cache[key]:
            hyp = cache[key]
        else:
            hyp = generate_hypothesis(failure)
            cache[key] = hyp
            with open(paths["llm_cache"], "w") as f:
                json.dump(cache, f)

        failure = dict(failure)
        failure["hypothesis"] = hyp
        failures_with_hyp.append(failure)

        done = i + 1
        if done % 10 == 0 or done == len(failures):
            print(f"  [{done}/{len(failures)}] idx={failure['image_idx']} {failure['true_class']}→{failure['predicted_class']} — {hyp[:80]}...")

    with open(paths["hyp"], "w") as f:
        json.dump(failures_with_hyp, f, indent=2)
    print(f"\nSaved to {paths['hyp']}")
    return failures_with_hyp


# ─── Step 4: Cluster ──────────────────────────────────────────────────────────
def cluster_hypotheses(failures_with_hyp: list):
    print("\n" + "=" * 60)
    print("STEP 4 — Clustering hypotheses")
    print("=" * 60)

    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [f["hypothesis"] for f in failures_with_hyp]
    print(f"Embedding {len(texts)} hypotheses...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)

    best_k, best_score = 3, -1
    for k in range(3, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        score = silhouette_score(embeddings, km.fit_predict(embeddings))
        print(f"  k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"\nOptimal clusters: {best_k} (silhouette={best_score:.3f})")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(embeddings)
    for i, f in enumerate(failures_with_hyp):
        f["cluster"] = int(cluster_labels[i])

    return failures_with_hyp, embeddings, km_final, best_k, best_score


# ─── Step 5: Label clusters ───────────────────────────────────────────────────
def label_cluster(hypotheses: list[str]) -> str:
    examples = "\n".join(f"- {h[:200]}" for h in hypotheses[:5])
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": f"""These are failure hypotheses from an image classifier grouped together:

{examples}

In one short phrase (5-8 words), what do these failures have in common?
Output only the phrase, nothing else."""},
            {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
        ],
        options={"temperature": 0.1, "num_predict": 50},
        stream=False,
    )
    return _strip_think(response.message.content or "").strip()


def label_clusters(failures_with_hyp: list, embeddings: np.ndarray, km: KMeans, best_k: int):
    print("\n" + "=" * 60)
    print("STEP 5 — Labeling clusters")
    print("=" * 60)

    centroids = km.cluster_centers_
    cluster_data = []
    for cid in range(best_k):
        members = [(i, f) for i, f in enumerate(failures_with_hyp) if f["cluster"] == cid]
        member_embeddings = embeddings[[m[0] for m in members]]
        dists = np.linalg.norm(member_embeddings - centroids[cid], axis=1)
        sorted_members = [members[j] for j in np.argsort(dists)]

        central_hyps = [failures_with_hyp[m[0]]["hypothesis"] for m in sorted_members[:5]]
        cluster_label = label_cluster(central_hyps)
        print(f"  Cluster {cid}: \"{cluster_label}\" ({len(members)} samples)")

        rep_examples = [
            {
                "image_idx":       f["image_idx"],
                "true_class":      f["true_class"],
                "predicted_class": f["predicted_class"],
                "hypothesis":      f["hypothesis"],
            }
            for _, f in sorted_members[:3]
        ]
        cluster_data.append({
            "cluster_id":            cid,
            "label":                 cluster_label,
            "size":                  len(members),
            "percentage":            round(len(members) / len(failures_with_hyp) * 100, 1),
            "representative_examples": rep_examples,
        })
    return cluster_data


# ─── Step 6: Save analysis ────────────────────────────────────────────────────
def save_analysis(failures_with_hyp, cluster_data, per_class_acc, best_k, best_score, paths, experiment_id):
    conf_pairs = Counter((f["true_class"], f["predicted_class"]) for f in failures_with_hyp)
    analysis = {
        "experiment_id": experiment_id,
        "dataset":        "cifar10_resnet",
        "summary": {
            "total_test":        10000,
            "total_failures":    len(failures_with_hyp),
            "failure_rate":      round(len(failures_with_hyp) / 10000, 4),
            "per_class_accuracy": per_class_acc,
            "optimal_clusters":  best_k,
            "silhouette_score":  round(float(best_score), 4),
        },
        "confused_pairs": [
            {"true": t, "predicted": p, "count": cnt}
            for (t, p), cnt in conf_pairs.most_common(20)
        ],
        "clusters":  cluster_data,
        "failures":  failures_with_hyp,
    }
    with open(paths["analysis"], "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {paths['analysis']}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    compare = {}
    if os.path.exists(COMPARE_PATH):
        with open(COMPARE_PATH) as f:
            compare = json.load(f)
    compare[experiment_id] = {
        "experiment_id":   experiment_id,
        "dataset":         "cifar10_resnet",
        "total_failures":  len(failures_with_hyp),
        "failure_rate":    analysis["summary"]["failure_rate"],
        "per_class_accuracy": per_class_acc,
        "top_confused_pairs": analysis["confused_pairs"][:5],
        "optimal_clusters":  best_k,
        "silhouette_score":  analysis["summary"]["silhouette_score"],
        "run_dir":         paths["dir"],
    }
    with open(COMPARE_PATH, "w") as f:
        json.dump(compare, f, indent=2)
    print(f"Comparison ledger updated at {COMPARE_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(experiment_id: str, augment: bool = False, epochs: int = 3,
         run_llm: bool = True, test_mode: bool = False, llm_samples: int = 0):
    paths = run_paths(experiment_id)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_id}  (dir: {paths['dir']})")
    print(f"{'='*60}")

    # Step 1 — Train or load
    if os.path.exists(paths["model"]):
        print(f"Found existing model, loading from {paths['model']}...")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(paths["model"], map_location=DEVICE))
        model = model.to(DEVICE)
        # Recompute per_class_acc from training log if available
        if os.path.exists(paths["training_log"]):
            with open(paths["training_log"]) as f:
                log = json.load(f)
            per_class_acc = log[-1]["per_class_val_acc"]
        else:
            per_class_acc = {}
    else:
        model, per_class_acc = train_model(paths, augment=augment, epochs=epochs)

    # Step 2 — Collect failures
    if os.path.exists(paths["failures"]) and os.path.exists(paths["predictions"]):
        print(f"\nFound cached failures/predictions, loading...")
        with open(paths["failures"])    as f: failures = json.load(f)
        with open(paths["predictions"]) as f: predictions = json.load(f)
        print(f"Loaded {len(failures)} failures")
    else:
        failures = collect_failures(model, paths)
        with open(paths["predictions"]) as f:
            predictions = json.load(f)

    if test_mode:
        print("\n*** TEST MODE: 2 failures only ***")
        failures = failures[:2]

    if llm_samples and not test_mode:
        from collections import defaultdict
        import random
        per_class: dict = defaultdict(list)
        for f in failures:
            per_class[f["true_class"]].append(f)
        sampled = []
        for cls, items in per_class.items():
            sampled.extend(random.sample(items, min(llm_samples, len(items))))
        print(f"\nSampled {len(sampled)} failures ({llm_samples}/class) for LLM hypotheses")
        failures = sampled

    if not run_llm:
        print("\nSkipping LLM steps (--no-llm). Done.")
        return

    # Step 3 — Hypotheses
    if not test_mode and os.path.exists(paths["hyp"]):
        print(f"\nFound {paths['hyp']}, loading...")
        with open(paths["hyp"]) as f:
            failures_with_hyp = json.load(f)
        missing = [f for f in failures_with_hyp if not f.get("hypothesis")]
        if missing:
            print(f"{len(missing)} missing hypotheses, generating...")
            failures_with_hyp = generate_hypotheses(failures_with_hyp, paths)
        else:
            print(f"All {len(failures_with_hyp)} hypotheses loaded")
    else:
        failures_with_hyp = generate_hypotheses(failures, paths)

    # Step 4 — Cluster
    if test_mode:
        for i, f in enumerate(failures_with_hyp):
            f["cluster"] = i % 2
        print("\n=== TEST COMPLETE ===")
        for f in failures_with_hyp:
            print(f"  idx={f['image_idx']} {f['true_class']}→{f['predicted_class']}: {f['hypothesis']}")
        return

    failures_with_hyp, embeddings, km, best_k, best_score = cluster_hypotheses(failures_with_hyp)

    # Step 5 — Label clusters
    cluster_data = label_clusters(failures_with_hyp, embeddings, km, best_k)

    # Step 6 — Save
    save_analysis(failures_with_hyp, cluster_data, per_class_acc, best_k, best_score, paths, experiment_id)


if __name__ == "__main__":
    args = sys.argv[1:]
    test_mode  = "--test"     in args
    no_augment = "--no-augment" in args
    no_llm     = "--no-llm"   in args

    epochs = 3
    llm_samples = 0
    for a in args:
        if a.startswith("--epochs="):
            epochs = int(a.split("=", 1)[1])
        if a.startswith("--llm-samples="):
            llm_samples = int(a.split("=", 1)[1])

    run_name = None
    for a in args:
        if a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
    if run_name is None:
        print("ERROR: --run-name=<experiment_id> is required")
        print("Examples:")
        print("  uv run python train_cifar_resnet.py --run-name=resnet_baseline --no-augment --no-llm")
        print("  uv run python train_cifar_resnet.py --run-name=resnet_baseline --llm-samples=15")
        print("  uv run python train_cifar_resnet.py --run-name=resnet_test --test")
        sys.exit(1)

    main(
        experiment_id=run_name,
        augment=not no_augment,
        epochs=epochs,
        run_llm=not no_llm,
        test_mode=test_mode,
        llm_samples=llm_samples,
    )
