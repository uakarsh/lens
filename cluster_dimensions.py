"""
Dimensional clustering of LLM hypotheses.

Each failure hypothesis has 3 parts:
  1. DATA ISSUE   — is this a labeling/annotation problem?
  2. VISUAL CAUSE — what visual property caused the confusion?
  3. FIX          — what augmentation or data fix would help?

This script clusters each dimension independently using BAAI/bge-base-en-v1.5,
then writes a dimensional_clusters.json alongside the run's analysis.json.

Usage:
  uv run python cluster_dimensions.py --run-name=resnet_baseline
  uv run python cluster_dimensions.py --run-name=resnet_baseline --k=5
"""

import json
import os
import re
import sys

from collections import Counter

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RUNS_DIR = "runs"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

DIMENSIONS = [
    {"key": "data_issue",   "section": "1. DATA ISSUE",   "label": "Data Issue",   "color": "#f9c74f"},
    {"key": "visual_cause", "section": "2. VISUAL CAUSE", "label": "Visual Cause", "color": "#7fb3f5"},
    {"key": "fix",          "section": "3. FIX",          "label": "Fix",          "color": "#4caf50"},
]


# ─── Parse hypothesis into sections ──────────────────────────────────────────

# Boilerplate phrases that appear in almost every section — strip before embedding
# so clusters form on the actual specific detail, not the shared opener.
_BOILERPLATE = re.compile(
    r"(this is an ambiguous image that a human could also get wrong\.?|"
    r"no,? this is not a labeling mistake or ambiguous image\.?|"
    r"the specific visual property causing confusion is|"
    r"to (help the model|address this)|"
    r"add more training examples of)",
    re.IGNORECASE,
)


def parse_hypothesis(text: str) -> dict[str, str]:
    """Split a 3-part hypothesis into its component sections, stripping boilerplate openers."""
    result = {}
    for i, dim in enumerate(DIMENSIONS):
        next_dim = DIMENSIONS[i + 1] if i + 1 < len(DIMENSIONS) else None
        start = re.search(dim["section"], text, re.IGNORECASE)
        if start is None:
            result[dim["key"]] = text
            continue
        end = re.search(next_dim["section"], text, re.IGNORECASE) if next_dim else None
        chunk = text[start.end(): end.start() if end else len(text)]
        chunk = re.sub(r"^[\s:?\n]+", "", chunk).strip()
        # Strip boilerplate so embeddings cluster on the specific detail
        chunk = _BOILERPLATE.sub("", chunk).strip()
        result[dim["key"]] = chunk
    return result


# ─── Cluster one dimension ────────────────────────────────────────────────────

_compress_cache: dict[str, str] = {}
_compress_cache_path: str = ""


def load_compress_cache(path: str):
    global _compress_cache, _compress_cache_path
    _compress_cache_path = path
    if os.path.exists(path):
        with open(path) as f:
            _compress_cache = json.load(f)
        print(f"  Loaded {len(_compress_cache)} cached compressions")


DATA_ISSUE_TAGS = ["annotation_issue", "ambiguous_image", "model_error"]


def compress_data_issue(text: str) -> tuple[str, str]:
    """
    Compress a verbose DATA ISSUE section.
    Returns (verdict_sentence, tag) where tag is one of:
      annotation_issue | ambiguous_image | model_error | corrupted_crop
    """
    key = text[:120]
    if key in _compress_cache:
        cached = _compress_cache[key]
        # cached may be a string (old format) or [verdict, tag]
        if isinstance(cached, list):
            return cached[0], cached[1]
        return cached, "model_error"

    response = ollama.chat(
        model="qwen3:latest",
        messages=[
            {
                "role": "user",
                "content": (
                    "Below is a data quality assessment for an image classifier failure.\n\n"
                    f"{text}\n\n"
                    "Output exactly 2 lines:\n"
                    "Line 1: ONE sentence (max 15 words) — what is the data quality issue? "
                    "Focus on the visual property, not the class names.\n"
                    "Line 2: ONE tag from this list exactly: "
                    "annotation_issue | ambiguous_image | model_error | corrupted_crop\n"
                    "  annotation_issue = wrong label in dataset\n"
                    "  ambiguous_image = image a human could also misclassify\n"
                    "  model_error = clean, correctly labeled image — pure model failure\n"
                    "Output only the 2 lines, nothing else."
                ),
            },
            {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
        ],
        options={"temperature": 0.1, "num_predict": 60},
        stream=False,
    )
    raw = response.message.content or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    lines = [l.strip().strip("*").strip() for l in raw.splitlines() if l.strip()]

    verdict = lines[0] if lines else text[:100]
    tag = "model_error"
    if len(lines) >= 2:
        for t in DATA_ISSUE_TAGS:
            if t in lines[1].lower():
                tag = t
                break

    _compress_cache[key] = [verdict, tag]
    if _compress_cache_path:
        with open(_compress_cache_path, "w") as f:
            json.dump(_compress_cache, f)
    return verdict, tag


def cluster_dimension(texts: list[str], embed_model, force_k: int = 0) -> tuple[np.ndarray, KMeans, int, float]:
    print(f"  Embedding {len(texts)} texts...")
    embeddings = embed_model.encode(texts, show_progress_bar=False)

    if force_k:
        k_range = [force_k]
    else:
        k_range = range(15, min(31, len(texts)))

    best_k, best_score = 3, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        print(f"    k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score, best_k = score, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_final.fit(embeddings)
    return embeddings, km_final, best_k, best_score


# ─── Label a cluster via LLM ─────────────────────────────────────────────────

DIM_FOCUS = {
    "Data Issue": (
        "These are 'data quality' assessments. Write a label that captures the SPECIFIC data problem shared "
        "by these examples — e.g. 'ship hull cropped to look like car', 'dog-deer confusion from background', "
        "'sail texture matches fur'. Be specific to what you see, not a generic category like 'model error'."
    ),
    "Visual Cause": (
        "These describe visual properties that caused misclassification. Write a label that names the SPECIFIC "
        "visual pattern — e.g. 'dark water background absorbs hull edges', 'sail folds mimic fur texture', "
        "'bow crop removes hull context'. Avoid repeating the same label for different clusters."
    ),
    "Fix": (
        "These are suggested fixes. Write a label naming the SPECIFIC fix strategy — e.g. "
        "'random crop augmentation for ships', 'background color jitter for animals', "
        "'add side-view ship training examples'. Each cluster should have a distinct, actionable label."
    ),
}


def label_cluster(texts: list[str], dimension_label: str, used_labels: list[str]) -> str:
    examples = "\n".join(f"- {t[:250]}" for t in texts[:5])
    focus = DIM_FOCUS.get(dimension_label, "")
    avoid = ""
    if used_labels:
        avoid = f"\nAlready used labels (do NOT repeat these): {', '.join(repr(l) for l in used_labels)}\n"
    response = ollama.chat(
        model="qwen3:latest",
        messages=[
            {
                "role": "user",
                "content": (
                    f"These are '{dimension_label}' notes from image classifier failure diagnoses:\n\n"
                    f"{examples}\n\n"
                    f"{focus}\n"
                    f"{avoid}"
                    f"Write a specific 4-7 word label for what these share. Output only the label."
                ),
            },
            {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
        ],
        options={"temperature": 0.3, "num_predict": 60},
        stream=False,
    )
    raw = response.message.content or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for line in raw.splitlines():
        line = line.strip().strip("*").strip()
        if line:
            return line
    return raw


# ─── Main ────────────────────────────────────────────────────────────────────

def main(run_name: str, force_k: int = 0, min_cluster_size: int = 8):
    hyp_path = os.path.join(RUNS_DIR, run_name, "failures_with_hypotheses.json")
    out_path  = os.path.join(RUNS_DIR, run_name, "dimensional_clusters.json")

    if not os.path.exists(hyp_path):
        print(f"ERROR: {hyp_path} not found. Run the LLM step first.")
        sys.exit(1)

    with open(hyp_path) as f:
        failures = json.load(f)

    # Filter to failures that actually have hypotheses
    failures = [f for f in failures if f.get("hypothesis", "").strip()]
    print(f"Loaded {len(failures)} failures with hypotheses")

    # Parse each hypothesis into 3 sections
    for f in failures:
        f["_sections"] = parse_hypothesis(f["hypothesis"])

    compress_cache_path = os.path.join(RUNS_DIR, run_name, "compress_cache.json")
    load_compress_cache(compress_cache_path)

    print(f"\nLoading embed model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    result = {"run_name": run_name, "dimensions": {}}

    for dim in DIMENSIONS:
        key   = dim["key"]
        label = dim["label"]
        print(f"\n{'='*60}")
        print(f"Clustering dimension: {label}")
        print(f"{'='*60}")

        raw_texts = [f["_sections"].get(key, f["hypothesis"]) for f in failures]

        # For data_issue: compress each section to a crisp verdict before embedding
        # so clusters form on data quality type, not class confusion pairs.
        if key == "data_issue":
            print(f"  Compressing {len(raw_texts)} DATA ISSUE sections...")
            texts = []
            for i, t in enumerate(raw_texts):
                verdict, tag = compress_data_issue(t)
                texts.append(verdict)
                failures[i]["data_issue_tag"] = tag
                failures[i]["data_issue_verdict"] = verdict
                if i < 3:
                    print(f"    [{i}] [{tag}] {verdict}")
            print(f"  ...done")
            # Print tag distribution
            from collections import Counter
            tag_counts = Counter(f.get("data_issue_tag", "?") for f in failures)
            print(f"  Tag distribution: {dict(tag_counts)}")
        else:
            texts = raw_texts
        embeddings, km, best_k, best_score = cluster_dimension(texts, embed_model, force_k)
        cluster_assignments = km.predict(embeddings)

        print(f"\n  Optimal k={best_k} (silhouette={best_score:.3f})")
        print("  Labeling clusters via LLM...")

        centroids = km.cluster_centers_
        clusters = []
        skipped = 0
        for cid in range(best_k):
            member_idxs = [i for i, c in enumerate(cluster_assignments) if c == cid]
            if len(member_idxs) < min_cluster_size:
                skipped += 1
                print(f"    Cluster {cid}: skipped (size {len(member_idxs)} < min {min_cluster_size})")
                continue
            member_embs = embeddings[member_idxs]
            dists = np.linalg.norm(member_embs - centroids[cid], axis=1)
            sorted_idxs = [member_idxs[j] for j in np.argsort(dists)]

            # Sample up to 10 members: 5 closest to centroid + 5 random for variety
            import random as _random
            centroid_idxs = sorted_idxs[:5]
            random_idxs = _random.sample(member_idxs, min(5, len(member_idxs)))
            sample_idxs = list(dict.fromkeys(centroid_idxs + random_idxs))  # dedup, preserve order
            central_examples = []
            for i in sample_idxs:
                f = failures[i]
                true_cls = f.get("true_class", f.get("true_label", "?"))
                pred_cls = f.get("predicted_class", f.get("predicted_label", "?"))
                # Use raw_texts for labeling (full detail), not the stripped verdict-only texts
                label_text = raw_texts[i] if key == "data_issue" else texts[i]
                central_examples.append(f"{true_cls}→{pred_cls}: {label_text[:150]}")
            used = [c["label"] for c in clusters]
            cluster_label = label_cluster(central_examples, label, used)
            print(f"    Cluster {cid}: \"{cluster_label}\" ({len(member_idxs)} failures)")

            rep_examples = []
            for i in sorted_idxs[:3]:
                f = failures[i]
                rep_examples.append({
                    "image_idx":       f["image_idx"],
                    "true_class":      f.get("true_class", f.get("true_label")),
                    "predicted_class": f.get("predicted_class", f.get("predicted_label")),
                    "section_text":    texts[i][:300],
                })

            clusters.append({
                "cluster_id":              cid,
                "label":                   cluster_label,
                "size":                    len(member_idxs),
                "percentage":              round(len(member_idxs) / len(failures) * 100, 1),
                "representative_examples": rep_examples,
            })

        # Write cluster assignment back to each failure
        for i, f in enumerate(failures):
            f[f"cluster_{key}"] = int(cluster_assignments[i])

        result["dimensions"][key] = {
            "label":          label,
            "color":          dim["color"],
            "best_k":         best_k,
            "silhouette":     round(float(best_score), 4),
            "clusters":       clusters,
        }

    # Attach actionable data issue summary to result
    ACTIONABLE_TAGS = {"annotation_issue", "ambiguous_image"}
    actionable = [
        {
            "image_idx":       f["image_idx"],
            "true_class":      f.get("true_class", f.get("true_label")),
            "predicted_class": f.get("predicted_class", f.get("predicted_label")),
            "tag":             f.get("data_issue_tag"),
            "verdict":         f.get("data_issue_verdict", ""),
        }
        for f in failures if f.get("data_issue_tag") in ACTIONABLE_TAGS
    ]
    result["data_issues"] = {
        "total_actionable": len(actionable),
        "tag_counts": dict(Counter(a["tag"] for a in actionable)),
        "failures": actionable,
    }

    # Save dimensional clusters file
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDimensional clusters saved to {out_path}")

    # Also patch failures_with_hypotheses.json with the new cluster keys
    with open(hyp_path, "w") as f:
        # Remove internal _sections before saving
        for fail in failures:
            fail.pop("_sections", None)
        json.dump(failures, f, indent=2)
    print(f"Updated {hyp_path} with per-dimension cluster assignments")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dim_key, dim_data in result["dimensions"].items():
        print(f"\n  {dim_data['label']} ({dim_data['best_k']} clusters, silhouette={dim_data['silhouette']}):")
        for c in dim_data["clusters"]:
            print(f"    [{c['cluster_id']}] {c['label']:40s}  {c['size']:4d} failures  ({c['percentage']}%)")

    # ── Data issue spotlight: only failures worth acting on ──────────────────
    ACTIONABLE_TAGS = {"annotation_issue", "ambiguous_image"}
    actionable = [f for f in failures if f.get("data_issue_tag") in ACTIONABLE_TAGS]
    if actionable:
        from collections import Counter
        print(f"\n{'='*60}")
        print(f"DATA ISSUES WORTH ACTING ON ({len(actionable)} / {len(failures)} failures)")
        print(f"{'='*60}")
        tag_counts = Counter(f["data_issue_tag"] for f in actionable)
        for tag, cnt in tag_counts.most_common():
            pct = round(cnt / len(failures) * 100, 1)
            print(f"  {tag:20s}  {cnt:4d} failures  ({pct}%)")
        print(f"\n  Top examples:")
        for f in actionable[:5]:
            true_cls = f.get("true_class", f.get("true_label", "?"))
            pred_cls = f.get("predicted_class", f.get("predicted_label", "?"))
            tag  = f.get("data_issue_tag", "?")
            verdict = f.get("_compressed_verdict", "")
            print(f"    [{tag}] {true_cls}→{pred_cls}  idx={f['image_idx']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    run_name = None
    force_k = 0
    min_cluster_size = 15
    for a in args:
        if a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
        if a.startswith("--k="):
            force_k = int(a.split("=", 1)[1])
        if a.startswith("--min-size="):
            min_cluster_size = int(a.split("=", 1)[1])

    if not run_name:
        print("ERROR: --run-name=<name> required")
        print("Usage: uv run python cluster_dimensions.py --run-name=resnet_baseline --min-size=8")
        sys.exit(1)

    main(run_name, force_k, min_cluster_size)
