"""
Extract CIFAR-10 test images by index for manual hypothesis verification.

Usage:
  # Extract specific indices
  uv run python extract_images.py --indices 5466 1685 4476 1506

  # Extract all 150 sampled failures
  uv run python extract_images.py --all-failures

  # Extract with hypothesis printed to stdout
  uv run python extract_images.py --indices 5466 --show-hypothesis

Images are saved to: runs/resnet_baseline/extracted_images/
Filename: {idx}_{true}_{predicted}.png
"""

import argparse
import json
import pickle
import struct
from pathlib import Path

import numpy as np
from PIL import Image

RUNS_DIR = Path("runs/resnet_baseline")
OUT_DIR = RUNS_DIR / "extracted_images"
CIFAR_DIR = Path("cifar-10-batches-py")
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def load_cifar10_test():
    with open(CIFAR_DIR / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (10000, 32, 32, 3)
    labels = d[b"labels"]
    return images, labels


def extract(indices, show_hypothesis=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = load_cifar10_test()

    with open(RUNS_DIR / "failures_with_hypotheses.json") as f:
        hyp_data = {str(d["image_idx"]): d for d in json.load(f)}

    for idx in indices:
        img_arr = images[idx]
        img = Image.fromarray(img_arr.astype(np.uint8))
        img = img.resize((128, 128), Image.NEAREST)  # upscale for visibility

        entry = hyp_data.get(str(idx), {})
        true_class = entry.get("true_class", CLASSES[labels[idx]])
        pred_class = entry.get("predicted_class", "unknown")
        conf = entry.get("confidence_predicted", 0)
        tag = entry.get("data_issue_tag", "")

        fname = f"{idx}_{true_class}_as_{pred_class}.png"
        img.save(OUT_DIR / fname)
        print(f"Saved: {OUT_DIR / fname}  [{tag}]  conf={conf:.3f}")

        if show_hypothesis and entry.get("hypothesis"):
            print(f"\n--- Hypothesis for {idx} ---")
            print(entry["hypothesis"])
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", help="Image indices to extract")
    parser.add_argument("--all-failures", action="store_true", help="Extract all 150 sampled failures")
    parser.add_argument("--show-hypothesis", action="store_true", help="Print hypothesis alongside image path")
    args = parser.parse_args()

    if args.all_failures:
        with open(RUNS_DIR / "failures_with_hypotheses.json") as f:
            data = json.load(f)
        indices = [d["image_idx"] for d in data]
    elif args.indices:
        indices = args.indices
    else:
        parser.print_help()
        return

    extract(indices, show_hypothesis=args.show_hypothesis)
    print(f"\nDone. {len(indices)} images in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
