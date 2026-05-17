# Lens — Model Error Analysis Platform

Lens is a human-in-the-loop error analysis tool for image classifiers. It trains a model, collects failures, generates per-failure hypotheses using a vision-language model, clusters those hypotheses across three independent dimensions, and surfaces the results in a review UI and dashboard.

## What it does

1. **Train** — fine-tune a model on a dataset, track per-class val loss every epoch
2. **Collect failures** — save every misclassification with top-3 predictions and confidence scores
3. **Hypothesize** — send each failure image to a local VLM (qwen3-vl via Ollama), get a structured 3-part diagnosis:
   - `DATA ISSUE` — is this a labeling error or ambiguous image?
   - `VISUAL CAUSE` — what specific visual property caused the confusion?
   - `FIX` — what augmentation or data change would help?
4. **Cluster** — embed and cluster each dimension independently using `BAAI/bge-base-en-v1.5` + KMeans, label clusters via LLM
5. **Review** — human annotates failures in a web UI, records agree/disagree/skip verdicts
6. **Dashboard** — summary view of all clusters, data quality issues, and actionable fixes

## Experiments

| Experiment | Dataset | Model | README |
|---|---|---|---|
| resnet_baseline | CIFAR-10 | ResNet-18 (pretrained) | [experiments/cifar_resnet/README.md](experiments/cifar_resnet/README.md) |

## Stack

- PyTorch + torchvision
- Ollama (local LLM inference) — `qwen3-vl:8b` for hypotheses, `qwen3:latest` for cluster labels
- SentenceTransformers — `BAAI/bge-base-en-v1.5` for hypothesis embeddings
- scikit-learn — KMeans + silhouette score
- FastAPI + vanilla JS — review UI and dashboard
- uv — Python environment management

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Pull Ollama models
ollama pull qwen3-vl:8b
ollama pull qwen3:latest
```

## Quick start

```bash
# Train + collect failures (no LLM)
uv run python train_cifar_resnet.py --run-name=my_run --no-augment --no-llm --epochs=3

# Generate hypotheses (samples 15 failures per class for speed)
uv run python train_cifar_resnet.py --run-name=my_run --no-augment --llm-samples=15

# Cluster across 3 dimensions
uv run python cluster_dimensions.py --run-name=my_run

# Launch review UI + dashboard
uv run python review_ui.py --run-name=my_run
# Open http://localhost:8000        (review)
# Open http://localhost:8000/dashboard  (cluster dashboard)
```

## Repo structure

```
train_cifar_resnet.py     # CIFAR-10 ResNet-18 pipeline (train → failures → hypotheses → cluster)
cluster_dimensions.py     # Dimensional clustering (data issue / visual cause / fix)
review_ui.py              # FastAPI review UI and dashboard
analyze_training_log.py   # Per-class loss analysis across epochs
train_cifar.py            # CIFAR-10 scratch CNN (ablation experiments)
train_and_analyze.py      # MNIST pipeline (original prototype)
runs/                     # Per-experiment results (training logs, analysis, feedback)
```
