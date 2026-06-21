from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss and test accuracy from history.csv")
    parser.add_argument("--history", required=True)
    parser.add_argument("--save_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    history_path = Path(args.history)
    df = pd.read_csv(history_path)
    save_path = Path(args.save_path) if args.save_path else history_path.with_name("training_curve.png")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["loss"], label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["acc"] * 100, linestyle="--", label="Test accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)

    plt.title(history_path.parent.name)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
