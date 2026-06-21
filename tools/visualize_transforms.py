from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import function.function as function
from CWRU.CWRU_dataset import CWRU


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize six signal-to-image transforms")
    parser.add_argument("--data_path", default="./CWRU")
    parser.add_argument("--save_path", default="assets/transform_comparison.png")
    parser.add_argument("--sample_index", type=int, default=None, help="If omitted, first fault sample is used")
    return parser.parse_args()


def main():
    args = parse_args()
    data = CWRU(1, ["12DriveEndFault"], ["1772", "1750", "1730"], 2048, args.data_path)
    x = data.X_train.astype(np.float32).reshape([-1, 4096])
    y = data.y_train
    idx = args.sample_index if args.sample_index is not None else int(np.where(y > 0)[0][0])
    signal = x[idx:idx + 1]

    transforms = ["SP", "GAF", "MTF", "RP", "SC", "GS"]
    resolutions = [128, 64]

    fig, axes = plt.subplots(len(resolutions), len(transforms), figsize=(18, 6))
    for r_i, res in enumerate(resolutions):
        for c_i, name in enumerate(transforms):
            img = function.get_transform(name)(signal, res=res).squeeze().numpy()
            ax = axes[r_i, c_i]
            ax.imshow(img, cmap="viridis", origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(name, fontsize=16, fontweight="bold")
            if c_i == 0:
                ax.set_ylabel(f"{res}×{res}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
