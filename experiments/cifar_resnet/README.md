# Experiment: CIFAR-10 ResNet-18 Baseline

Fine-tuning pretrained ResNet-18 on CIFAR-10, followed by full Lens error analysis.

## Model

- **Architecture**: ResNet-18 (ImageNet pretrained weights)
- **Head**: `Linear(512, 10)` replacing the original FC
- **Optimizer**: AdamW — FC at lr=1e-3, backbone at lr=1e-4, weight_decay=1e-4
- **Scheduler**: CosineAnnealingLR
- **Augmentation**: none (baseline)
- **Epochs**: 3

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| airplane | 83.1% | 81.8% | 82.5% |
| automobile | 86.5% | 88.1% | 87.3% |
| bird | 75.9% | 71.1% | 73.4% |
| **cat** | **61.9%** | **61.9%** | **61.9%** |
| deer | 73.8% | 77.2% | 75.5% |
| **dog** | **70.8%** | **69.2%** | **70.0%** |
| frog | 81.5% | 87.1% | 84.2% |
| horse | 84.3% | 80.8% | 82.5% |
| ship | 86.8% | 89.7% | 88.2% |
| truck | 85.6% | 83.4% | 84.5% |

Cat (61.9%) and dog (70.0%) are the weakest classes — both show val loss rising after epoch 2, indicating boundary ambiguity rather than insufficient capacity.

## Error analysis

**150 failures sampled** (15 per class) for LLM hypothesis generation using `qwen3-vl:8b`.

### Data quality issues (84 / 150 failures)

| Tag | Count |
|---|---|
| ambiguous_image | 68 |
| annotation_issue | 16 |

68 failures are images a human would also struggle with — mostly due to cropping that removes key features or background textures that bleed into the object. 16 appear to be outright labeling errors.

### Visual cause clusters (top 3)

| Cluster | Failures | Pattern |
|---|---|---|
| texture similarity with background | 9 | Background texture matches object texture closely |
| top-down crop mimics airplane underside | 8 | Ship hull cropped to expose only lower portion |
| partial occlusion by dark textured elements | 14 | Dark foreground objects hide discriminative features |

### Fix clusters (top 3)

| Cluster | Failures | Fix |
|---|---|---|
| Tail Section Cropping Augmentation | 7 | Random crops that cut tails/rear sections |
| Partial Head Occlusion Augmentation | 10 | Simulate head occlusion in training |
| Background Texture Augmentation | 7 | Color jitter + texture aug for frog/animal classes |

## Run artifacts

| File | Description |
|---|---|
| `runs/resnet_baseline/training_log.json` | Per-class val loss + acc every epoch |
| `runs/resnet_baseline/analysis.json` | Flat cluster assignments + confused pairs |
| `runs/resnet_baseline/dimensional_clusters.json` | 3-axis cluster breakdown (data / visual / fix) |
| `runs/resnet_baseline/compress_cache.json` | LLM-compressed data issue verdicts |
| `runs/resnet_baseline/human_feedback.json` | Human annotations from review UI |

## Reproduce

```bash
# Train (skips if model.pth exists)
uv run python train_cifar_resnet.py --run-name=resnet_baseline --no-augment --no-llm --epochs=3

# Generate hypotheses
uv run python train_cifar_resnet.py --run-name=resnet_baseline --no-augment --llm-samples=15

# Dimensional clustering
uv run python cluster_dimensions.py --run-name=resnet_baseline

# Review UI
uv run python review_ui.py --run-name=resnet_baseline
```

## Next steps

- Train iteration 2 with targeted augmentations from fix clusters (head occlusion, background texture, crop aug)
- Compare resnet_baseline vs resnet_iter2 per-class F1
- Investigate the 16 annotation_issue failures — likely candidates for relabeling or removal
