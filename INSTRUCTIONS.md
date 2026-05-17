# Lens — MNIST Error Analysis POC
## Instructions for Claude Code

---

## What is Lens?

Lens is an open-source **model error analysis framework**. The core idea:

> Most evaluation frameworks tell you your model's score. Lens tells you *why* it failed — and what to do about it.

The framework has four steps:
1. **Detect** — find which samples the model got wrong
2. **Diagnose** — use an LLM to generate a free-form hypothesis for each failure
3. **Cluster** — embed all hypotheses and cluster them to discover patterns
4. **Report** — surface the emergent taxonomy with representative examples

**Critical design principle:** Lens does NOT impose a fixed hypothesis taxonomy upfront. The taxonomy emerges from the data through clustering. This is what separates Lens from rule-based error analysis.

---

## What we are building tonight

A proof of concept of Lens applied to MNIST classification errors.

### The user story

> "I trained an MNIST classifier. It gets 98% accuracy. I don't know why the remaining 2% fail. Are they hard images? Specific digit confusion? Systematic patterns I can fix?"

Lens answers that question — without the developer having to guess the failure modes in advance.

---

## Step 1 — Train a CNN on MNIST

Simple architecture, PyTorch:
- 2 conv layers (32 and 64 filters) + 2 FC layers
- ReLU activations, max pooling, dropout 0.5
- Train 5 epochs on MNIST training set (60k images)
- Evaluate on test set (10k images)
- Save model to `lens_mnist_model.pth`
- Print per-class accuracy

---

## Step 2 — Collect failures with image statistics

For every misclassified test image, extract:

```python
failure = {
    "image_idx": int,
    "true_label": int,
    "predicted_label": int,
    "confidence_predicted": float,   # softmax prob of predicted class
    "confidence_true": float,        # softmax prob of true class
    "image_stats": {
        # All computable from raw 28x28 pixel tensor
        "mean_intensity": float,     # mean pixel value (pixels normalized 0-1)
        "std_intensity": float,      # std of pixel values
        "nonzero_ratio": float,      # fraction of pixels above threshold 0.1
        "nonzero_count": int,        # number of lit pixels
        "center_of_mass_x": float,   # np.average(col_indices, weights=pixel_values), normalized 0-1
        "center_of_mass_y": float,   # np.average(row_indices, weights=pixel_values), normalized 0-1
        "bbox_rmin": int,            # min row with nonzero pixel
        "bbox_rmax": int,            # max row with nonzero pixel
        "bbox_cmin": int,            # min col with nonzero pixel
        "bbox_cmax": int,            # max col with nonzero pixel
        "bbox_height": int,          # rmax - rmin
        "bbox_width": int,           # cmax - cmin
        "aspect_ratio": float,       # bbox_height / bbox_width (or 1.0 if width=0)
    }
}
```

Save to `lens_mnist_failures.json`. Print confusion matrix.

---

## Step 3 — LLM hypothesis generation (open-ended, no fixed classes)

For each failure, call Ollama with qwen3:8b to generate a **free-form hypothesis**.

Do NOT give the model a list of hypotheses to choose from.
Do NOT constrain the output to fixed categories.
Let the model reason freely from the statistics.

```python
import ollama
import json

def generate_hypothesis(failure: dict) -> str:
    s = failure["image_stats"]
    prompt = f"""A handwritten digit classifier made this mistake.

True digit: {failure['true_label']}
Predicted digit: {failure['predicted_label']}
Model confidence in wrong prediction: {failure['confidence_predicted']:.1%}
Model confidence in correct label: {failure['confidence_true']:.1%}

Image statistics (28x28 grayscale, pixel values 0-1):
- Mean pixel intensity: {s['mean_intensity']:.3f}
- Pixel std deviation: {s['std_intensity']:.3f}
- Fraction of lit pixels (>0.1 threshold): {s['nonzero_ratio']:.3f}
- Center of mass: ({s['center_of_mass_x']:.2f}, {s['center_of_mass_y']:.2f}) where (0,0)=top-left (1,1)=bottom-right
- Digit bounding box: rows {s['bbox_rmin']}-{s['bbox_rmax']}, cols {s['bbox_cmin']}-{s['bbox_cmax']} (out of 0-27)
- Bounding box size: {s['bbox_height']}h x {s['bbox_width']}w pixels
- Aspect ratio (h/w): {s['aspect_ratio']:.2f}

In 2-3 sentences, explain the most likely reason this classifier made this mistake.
Be specific and grounded in the statistics provided.
Do not speculate beyond what the numbers support.
Do not mention that you are an AI."""

    response = ollama.chat(
        model='qwen3:8b',
        messages=[
            {
                'role': 'system',
                'content': (
                    "You are an expert at diagnosing image classifier failures. "
                    "Given statistics about a misclassified handwritten digit, "
                    "generate a specific, grounded hypothesis about why the "
                    "classifier failed. Base your reasoning only on the "
                    "provided statistics."
                )
            },
            {'role': 'user', 'content': prompt}
        ],
        options={'temperature': 0.1, 'num_predict': 150},
        think=False,
        stream=False
    )
    return response['message']['content'].strip()
```

**Caching — mandatory:**
```python
CACHE_FILE = "lens_mnist_llm_cache.json"
# key = str(failure['image_idx'])
# Load on start, save after every new entry
# If key exists in cache, skip LLM call
```

Save hypotheses back into the failures list:
```python
failure['hypothesis'] = generated_hypothesis_text
```

Save updated failures to `lens_mnist_failures_with_hypotheses.json`.

---

## Step 4 — Cluster hypotheses to discover taxonomy

Embed all hypothesis texts and cluster them.
Do NOT hardcode the number of clusters.
Use silhouette score to find optimal k between 3 and 10.

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Embed hypotheses
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, fits in RAM
hypothesis_texts = [f['hypothesis'] for f in failures_with_hypotheses]
embeddings = embed_model.encode(hypothesis_texts, show_progress_bar=True)

# Find optimal k using silhouette score
best_k = 3
best_score = -1
for k in range(3, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"k={k}: silhouette={score:.3f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nOptimal clusters: {best_k}")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# Assign cluster to each failure
for i, failure in enumerate(failures_with_hypotheses):
    failure['cluster'] = int(cluster_labels[i])
```

---

## Step 5 — Label clusters with LLM

For each cluster, pick the 5 most central examples (closest to centroid).
Ask the LLM to summarize what they have in common.

```python
def label_cluster(cluster_hypotheses: list[str]) -> str:
    examples = "\n".join([f"- {h}" for h in cluster_hypotheses[:5]])
    prompt = f"""These are failure hypotheses from a digit classifier that were grouped together:

{examples}

In one short phrase (5-8 words), what do these failures have in common?
Output only the phrase, nothing else."""

    response = ollama.chat(
        model='qwen3:8b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1, 'num_predict': 30},
        think=False,
        stream=False
    )
    return response['message']['content'].strip()
```

---

## Step 6 — Final report

Print to console:

```
============================================================
LENS ERROR ANALYSIS REPORT — MNIST
============================================================
Total test samples:  10,000
Total failures:      198
Failure rate:        1.98%

Most confused digit pairs:
  4 → 9:  28 times
  3 → 5:  19 times
  7 → 1:  16 times

Emergent failure taxonomy (discovered from data):
------------------------------------------------------------
Cluster 1 (41%): "visually similar digit pair confusion"
  Representative example:
    True: 4, Predicted: 9
    Hypothesis: "The digit 4 and 9 share similar upper loop structure..."

Cluster 2 (28%): "ambiguous low-confidence predictions"
  Representative example:
    True: 5, Predicted: 3
    Hypothesis: "Model shows low confidence (34%) suggesting..."

Cluster 3 (18%): "unusual stroke thickness or density"
  ...

Cluster 4 (13%): "off-center or rotated writing style"
  ...

------------------------------------------------------------
Actionable recommendations:
  Cluster 1 (41%): Add training examples of confusable pairs (4,9), (7,1)
  Cluster 2 (28%): Review these samples — possible labeling errors
  Cluster 3 (18%): Apply stroke augmentation during training
============================================================
```

---

## Step 7 — Save full analysis

```python
# lens_mnist_analysis.json
{
    "summary": {
        "total_test": 10000,
        "total_failures": int,
        "failure_rate": float,
        "per_class_accuracy": {"0": float, "1": float, ...},
        "optimal_clusters": int,
        "silhouette_score": float
    },
    "confused_pairs": [
        {"true": int, "predicted": int, "count": int}, ...
    ],
    "clusters": [
        {
            "cluster_id": int,
            "label": str,          # LLM-generated label
            "size": int,
            "percentage": float,
            "representative_examples": [...]  # 3 closest to centroid
        }, ...
    ],
    "failures": [...]  # full per-sample data including hypothesis and cluster
}
```

---

## File structure

```
lens-mnist/
  train_and_analyze.py         # single runnable script
  lens_mnist_model.pth         # saved after training
  lens_mnist_failures.json     # raw failures with image stats
  lens_mnist_llm_cache.json    # LLM response cache (resumable)
  lens_mnist_failures_with_hypotheses.json
  lens_mnist_analysis.json     # final complete report
  README.md                    # one paragraph + how to run
```

---

## Technical requirements

```
pip install torch torchvision sentence-transformers scikit-learn ollama
ollama pull qwen3:8b
ollama serve  # must be running
```

No GPU required. CPU training is fine for MNIST.

---

## Design principles

1. **Single script** — `python train_and_analyze.py` does everything
2. **Resumable** — LLM cache means interrupted runs pick up where they left off
3. **No fixed taxonomy** — hypotheses and clusters emerge from data
4. **Grounded** — LLM only reasons from provided statistics, no hallucination
5. **Actionable** — output tells developer what to do, not just what went wrong
6. **Self-contained** — no external APIs, runs completely locally

---

## Success criteria

Running `python train_and_analyze.py` should:
1. Train model, print per-epoch and per-class accuracy
2. Collect failures, print confusion matrix
3. Generate hypotheses with progress bar, using cache if available
4. Find optimal clusters, print silhouette scores
5. Print final report with emergent taxonomy
6. Save `lens_mnist_analysis.json`

The emergent taxonomy should make sense — clusters should be interpretable and different from each other. If all samples land in one cluster, something is wrong with the clustering step.

