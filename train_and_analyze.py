import base64
import io
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from PIL import Image
import ollama
from collections import Counter


class WhitePixelDropout:
    """Zero out a random fraction of the bright (foreground) pixels in a tensor."""
    def __init__(self, drop_prob: float = 0.12):
        self.drop_prob = drop_prob

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img is [1, H, W] after ToTensor, values in [0,1] pre-normalize
        # Work on a clone so we don't mutate the original
        img = img.clone()
        # Foreground mask: pixels brighter than 0.1 (stroke, not background)
        fg = img[0] > 0.1
        fg_indices = fg.nonzero(as_tuple=False)  # (N, 2)
        n_drop = int(len(fg_indices) * self.drop_prob)
        if n_drop > 0:
            chosen = fg_indices[torch.randperm(len(fg_indices))[:n_drop]]
            img[0, chosen[:, 0], chosen[:, 1]] = 0.0
        return img

OLLAMA_MODEL = "qwen3:latest"

# Loaded once before hypothesis generation; maps image_idx -> 28x28 float array
_test_images: dict[int, np.ndarray] = {}


def _load_test_images():
    if _test_images:
        return
    ds = datasets.MNIST(".", train=False, download=False, transform=transforms.ToTensor())
    for idx in range(len(ds)):
        img, _ = ds[idx]
        _test_images[idx] = img.squeeze().numpy()  # [0,1] float


def _img_to_b64(arr: np.ndarray) -> str:
    """Upscale 28x28 float array to 112x112 PNG and return base64 string."""
    uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(uint8, mode="L").resize((112, 112), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ─── Paths ───────────────────────────────────────────────────────────────────
RUNS_DIR = "runs"
COMPARE_PATH = "runs/compare.json"

DEVICE = torch.device("cpu")


def run_paths(run_name: str) -> dict:
    base = os.path.join(RUNS_DIR, run_name)
    os.makedirs(base, exist_ok=True)
    return {
        "dir":          base,
        "model":        os.path.join(base, "model.pth"),
        "failures":     os.path.join(base, "failures.json"),
        "predictions":  os.path.join(base, "predictions.json"),
        "llm_cache":    os.path.join(base, "llm_cache.json"),
        "hyp":          os.path.join(base, "failures_with_hypotheses.json"),
        "analysis":     os.path.join(base, "analysis.json"),
    }


# ─── Step 1: CNN Architecture ─────────────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_model(paths: dict, augment: bool = True):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        *([
            transforms.RandomResizedCrop(28, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=15, fill=0),
            WhitePixelDropout(drop_prob=0.12),
        ] if augment else []),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(".", train=True, download=True, transform=train_transform)
    test_ds = datasets.MNIST(".", train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MnistCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print(f"STEP 1 — Training CNN on MNIST (augment={augment})")
    print("=" * 60)

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/5", leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"  Epoch {epoch}/5 — avg loss: {total_loss / len(train_loader):.4f}")

    # Per-class accuracy
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for label, pred in zip(labels, preds):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    print("\nPer-class accuracy:")
    per_class_acc = {}
    for i in range(10):
        acc = class_correct[i] / class_total[i]
        per_class_acc[str(i)] = acc
        print(f"  Digit {i}: {acc:.1%}")

    torch.save(model.state_dict(), paths["model"])
    print(f"\nModel saved to {paths['model']}")
    return model, test_loader, per_class_acc


# ─── Step 2: Collect Failures ─────────────────────────────────────────────────
def collect_failures(model, paths: dict):
    print("\n" + "=" * 60)
    print("STEP 2 — Collecting failures")
    print("=" * 60)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST(".", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model.eval()
    failures = []
    all_predictions = []
    idx_offset = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu()
            for i in range(len(labels)):
                true = int(labels[i])
                pred = int(preds[i])
                conf_pred = float(probs[i][pred])
                conf_true = float(probs[i][true])
                all_predictions.append({
                    "image_idx": idx_offset + i,
                    "true_label": true,
                    "predicted_label": pred,
                    "correct": true == pred,
                    "confidence_predicted": conf_pred,
                    "confidence_true": conf_true,
                })
                if true != pred:
                    failures.append({
                        "image_idx": idx_offset + i,
                        "true_label": true,
                        "predicted_label": pred,
                        "confidence_predicted": conf_pred,
                        "confidence_true": conf_true,
                    })
            idx_offset += len(labels)

    conf_pairs = Counter((f["true_label"], f["predicted_label"]) for f in failures)
    print(f"\nTotal failures: {len(failures)} / 10,000  ({len(failures)/10000:.2%} error rate)")
    print("\nTop confused pairs:")
    for (t, p), cnt in conf_pairs.most_common(10):
        print(f"  {t} → {p}: {cnt} times")

    # Per-class precision / recall / F1
    tp = [0]*10; fp = [0]*10; fn = [0]*10; total = [0]*10
    for p in all_predictions:
        t, pred = p["true_label"], p["predicted_label"]
        total[t] += 1
        if t == pred:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[pred] += 1

    print(f"\n{'Digit':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 46)
    for i in range(10):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{i:>6} {prec:>10.1%} {rec:>8.1%} {f1:>8.1%} {total[i]:>9}")

    with open(paths["failures"], "w") as f:
        json.dump(failures, f)
    print(f"\nSaved failures to {paths['failures']}")

    with open(paths["predictions"], "w") as f:
        json.dump(all_predictions, f)
    print(f"Saved all predictions to {paths['predictions']}")

    return failures


# ─── Step 3: LLM Hypothesis Generation ───────────────────────────────────────
def generate_hypothesis(failure: dict) -> str:
    img_arr = _test_images.get(failure["image_idx"])
    img_b64 = _img_to_b64(img_arr) if img_arr is not None else None

    text_prompt = f"""A handwritten digit classifier made this mistake.

True digit: {failure['true_label']}
Predicted digit: {failure['predicted_label']}
Model confidence in wrong prediction: {failure['confidence_predicted']:.1%}
Model confidence in correct label: {failure['confidence_true']:.1%}

In 2-3 sentences, explain the most likely reason this classifier made this mistake.
Be specific and concise. Do not think out loud. Do not mention that you are an AI."""

    user_msg = {"role": "user", "content": text_prompt}

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at diagnosing image classifier failures. "
                    "You are given the actual digit image and its statistics. "
                    "Generate a specific, grounded hypothesis about why the "
                    "classifier failed. Base your reasoning on what you see in "
                    "the image and the provided statistics."
                ),
            },
            user_msg,
        ],
        options={"temperature": 0.1, "num_predict": 200},
        think=False,
        stream=False,
    )
    return response.message.content.strip()


def generate_hypotheses(failures: list, paths: dict) -> list:
    print("\n" + "=" * 60)
    print("STEP 3 — Generating LLM hypotheses")
    print("=" * 60)

    _load_test_images()

    cache = {}
    if os.path.exists(paths["llm_cache"]):
        with open(paths["llm_cache"]) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached hypotheses from {paths['llm_cache']}")

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
            print(f"  [{done}/{len(failures)}] idx={failure['image_idx']} — {hyp[:80]}...")

    with open(paths["hyp"], "w") as f:
        json.dump(failures_with_hyp, f, indent=2)
    print(f"\nSaved to {paths['hyp']}")
    return failures_with_hyp


# ─── Step 4: Cluster Hypotheses ───────────────────────────────────────────────
def cluster_hypotheses(failures_with_hyp: list):
    print("\n" + "=" * 60)
    print("STEP 4 — Clustering hypotheses")
    print("=" * 60)

    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [f["hypothesis"] for f in failures_with_hyp]
    print(f"Embedding {len(texts)} hypotheses...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)

    best_k = 3
    best_score = -1
    for k in range(3, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nOptimal clusters: {best_k} (silhouette={best_score:.3f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(embeddings)

    for i, failure in enumerate(failures_with_hyp):
        failure["cluster"] = int(cluster_labels[i])

    return failures_with_hyp, embeddings, km_final, best_k, best_score


# ─── Step 5: Label Clusters ───────────────────────────────────────────────────
def label_cluster(cluster_hypotheses_list: list[str]) -> str:
    examples = "\n".join([f"- {h}" for h in cluster_hypotheses_list[:5]])
    prompt = f"""These are failure hypotheses from a digit classifier that were grouped together:

{examples}

In one short phrase (5-8 words), what do these failures have in common?
Output only the phrase, nothing else."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 50},
        think=False,
        stream=False,
    )
    return response.message.content.strip()


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

        rep_examples = []
        for idx, f in sorted_members[:3]:
            rep_examples.append({
                "image_idx": f["image_idx"],
                "true_label": f["true_label"],
                "predicted_label": f["predicted_label"],
                "hypothesis": f["hypothesis"],
            })

        cluster_data.append({
            "cluster_id": cid,
            "label": cluster_label,
            "size": len(members),
            "percentage": round(len(members) / len(failures_with_hyp) * 100, 1),
            "representative_examples": rep_examples,
        })

    return cluster_data


# ─── Step 6: Final Report ─────────────────────────────────────────────────────
def print_report(failures_with_hyp: list, cluster_data: list, per_class_acc: dict):
    print("\n" + "=" * 60)
    print("LENS ERROR ANALYSIS REPORT — MNIST")
    print("=" * 60)

    total = 10000
    n_fail = len(failures_with_hyp)
    print(f"Total test samples:  {total:,}")
    print(f"Total failures:      {n_fail:,}")
    print(f"Failure rate:        {n_fail/total:.2%}")

    conf_pairs = Counter((f["true_label"], f["predicted_label"]) for f in failures_with_hyp)
    print("\nMost confused digit pairs:")
    for (t, p), cnt in conf_pairs.most_common(5):
        print(f"  {t} → {p}:  {cnt} times")

    print("\nEmergent failure taxonomy (discovered from data):")
    print("-" * 60)
    for c in sorted(cluster_data, key=lambda x: -x["size"]):
        pct = c["percentage"]
        label = c["label"]
        rep = c["representative_examples"][0]
        print(f"\nCluster {c['cluster_id']+1} ({pct:.0f}%): \"{label}\"")
        print(f"  Representative example:")
        print(f"    True: {rep['true_label']}, Predicted: {rep['predicted_label']}")
        print(f"    Hypothesis: \"{rep['hypothesis'][:100]}...\"")

    print("\n" + "-" * 60)
    print("Actionable recommendations:")
    for c in sorted(cluster_data, key=lambda x: -x["size"]):
        pct = c["percentage"]
        label = c["label"]
        if pct >= 30:
            action = "Add targeted training examples for most common confusable pairs"
        elif pct >= 20:
            action = "Review samples — possible labeling errors or ambiguous writing styles"
        else:
            action = "Apply augmentation (rotation, stroke width) to address variation"
        print(f"  Cluster {c['cluster_id']+1} ({pct:.0f}%): {action}")

    print("=" * 60)


# ─── Step 7: Save Analysis ────────────────────────────────────────────────────
def save_analysis(failures_with_hyp, cluster_data, per_class_acc, best_k, best_score, paths: dict, experiment_id: str):
    conf_pairs = Counter((f["true_label"], f["predicted_label"]) for f in failures_with_hyp)
    analysis = {
        "experiment_id": experiment_id,
        "summary": {
            "total_test": 10000,
            "total_failures": len(failures_with_hyp),
            "failure_rate": round(len(failures_with_hyp) / 10000, 4),
            "per_class_accuracy": per_class_acc,
            "optimal_clusters": best_k,
            "silhouette_score": round(float(best_score), 4),
        },
        "confused_pairs": [
            {"true": int(t), "predicted": int(p), "count": cnt}
            for (t, p), cnt in conf_pairs.most_common(20)
        ],
        "clusters": cluster_data,
        "failures": failures_with_hyp,
    }
    with open(paths["analysis"], "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {paths['analysis']}")

    # Update cross-run comparison ledger
    os.makedirs(RUNS_DIR, exist_ok=True)
    compare = {}
    if os.path.exists(COMPARE_PATH):
        with open(COMPARE_PATH) as f:
            compare = json.load(f)
    compare[experiment_id] = {
        "experiment_id": experiment_id,
        "total_failures": len(failures_with_hyp),
        "failure_rate": analysis["summary"]["failure_rate"],
        "per_class_accuracy": per_class_acc,
        "top_confused_pairs": analysis["confused_pairs"][:5],
        "optimal_clusters": best_k,
        "silhouette_score": analysis["summary"]["silhouette_score"],
        "run_dir": paths["dir"],
    }
    with open(COMPARE_PATH, "w") as f:
        json.dump(compare, f, indent=2)
    print(f"Comparison ledger updated at {COMPARE_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(experiment_id: str, test_mode: bool = False, augment: bool = True):
    paths = run_paths(experiment_id)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_id}  (dir: {paths['dir']})")
    print(f"{'='*60}")

    # Step 1 — Train / load model
    if os.path.exists(paths["model"]):
        print(f"Found existing model at {paths['model']}, loading...")
        model = MnistCNN().to(DEVICE)
        model.load_state_dict(torch.load(paths["model"], map_location=DEVICE))
        val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_ds = datasets.MNIST(".", train=False, download=True, transform=val_transform)
        class_correct = [0] * 10
        class_total = [0] * 10
        loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu()
                for label, pred in zip(labels, preds):
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        per_class_acc = {str(i): class_correct[i] / class_total[i] for i in range(10)}
    else:
        model, _, per_class_acc = train_model(paths, augment=augment)

    # Step 2 — Collect failures
    if os.path.exists(paths["failures"]):
        print(f"\nFound {paths['failures']}, loading...")
        with open(paths["failures"]) as f:
            failures = json.load(f)
        print(f"Loaded {len(failures)} failures")
    else:
        failures = collect_failures(model, paths)

    if test_mode:
        print("\n*** TEST MODE: running on 2 failures only ***")
        failures = failures[:2]

    # Step 3 — Hypotheses
    if not test_mode and os.path.exists(paths["hyp"]):
        print(f"\nFound {paths['hyp']}, loading...")
        with open(paths["hyp"]) as f:
            failures_with_hyp = json.load(f)
        missing = [f for f in failures_with_hyp if not f.get("hypothesis")]
        if missing:
            print(f"{len(missing)} failures missing hypotheses, generating...")
            failures_with_hyp = generate_hypotheses(failures_with_hyp, paths)
        else:
            print(f"All {len(failures_with_hyp)} hypotheses loaded")
    else:
        failures_with_hyp = generate_hypotheses(failures, paths)

    # Step 4 — Cluster
    if test_mode:
        print("\n*** TEST MODE: skipping clustering ***")
        for i, f in enumerate(failures_with_hyp):
            f["cluster"] = i % 2
        print("\n=== TEST COMPLETE ===")
        for f in failures_with_hyp:
            print(f"\n  idx={f['image_idx']}  true={f['true_label']}  pred={f['predicted_label']}")
            print(f"  hypothesis: {f['hypothesis']}")
        return

    failures_with_hyp, embeddings, km, best_k, best_score = cluster_hypotheses(failures_with_hyp)

    # Step 5 — Label clusters
    cluster_data = label_clusters(failures_with_hyp, embeddings, km, best_k)

    # Step 6 — Report
    print_report(failures_with_hyp, cluster_data, per_class_acc)

    # Step 7 — Save
    save_analysis(failures_with_hyp, cluster_data, per_class_acc, best_k, best_score, paths, experiment_id)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    test_mode = "--test" in args
    no_augment = "--no-augment" in args

    run_name = None
    for a in args:
        if a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
    if run_name is None:
        print("ERROR: --run-name=<experiment_id> is required")
        print("Example: uv run python train_and_analyze.py --run-name=baseline")
        print("         uv run python train_and_analyze.py --run-name=augmented")
        sys.exit(1)

    main(experiment_id=run_name, test_mode=test_mode, augment=not no_augment)
