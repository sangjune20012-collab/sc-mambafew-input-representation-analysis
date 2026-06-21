from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import function.function as function
from CWRU.CWRU_dataset import CWRU
from dataloader.dataloader import FewshotDataset
from net.new_proposed import MainNet


CLASS_NAMES = [
    "Normal",
    "0.007-Ball",
    "0.014-Ball",
    "0.021-Ball",
    "0.007-Inner",
    "0.014-Inner",
    "0.021-Inner",
    "0.007-Outer6",
    "0.014-Outer6",
    "0.021-Outer6",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SC-MambaFew input representation test on CWRU")
    parser.add_argument("--dataset", default="CWRU", choices=["CWRU"])
    parser.add_argument("--data_path", default="./CWRU")
    parser.add_argument("--transform", default="SP", choices=["SP", "SC", "GS", "RP", "MTF", "GAF"])
    parser.add_argument("--resolution", type=int, default=64, choices=[64, 128])
    parser.add_argument("--shot_num", type=int, default=1, choices=[1, 5])
    parser.add_argument("--training_samples_CWRU", type=int, default=None)
    parser.add_argument("--episode_num_test", type=int, default=75)
    parser.add_argument("--way_num_CWRU", type=int, default=10)
    parser.add_argument("--query_num", type=int, default=1)
    parser.add_argument("--weight", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--noise_db", type=float, default=None, help="If provided, add Gaussian noise to test set with this SNR dB")
    parser.add_argument("--test_iter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def prepare_cwru(args):
    window_size = 2048
    if args.training_samples_CWRU is None:
        args.training_samples_CWRU = 30 if args.shot_num == 1 else 150
    split = args.training_samples_CWRU // 30
    data = CWRU(split, ["12DriveEndFault"], ["1772", "1750", "1730"], window_size, args.data_path)

    test_x = data.X_test.astype(np.float32).reshape([750, 4096])
    if args.noise_db is not None:
        test_x = function.add_gaussian_noise_snr(test_x, args.noise_db, seed=args.seed)

    transform_fn = function.get_transform(args.transform)
    test_data = transform_fn(test_x, res=args.resolution)
    test_label = torch.from_numpy(data.y_test).long()
    return test_data, test_label


def save_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Row-normalized rate"},
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    function.seed_func(args.seed)
    device = torch.device(args.device)

    condition = "clean" if args.noise_db is None else f"noise_{args.noise_db:g}db"
    run_name = f"cwru_{args.shot_num}shot_{args.resolution}_{args.transform}_{condition}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data, test_label = prepare_cwru(args)
    h = args.resolution // 4
    net = MainNet(h=h, w=h).to(device)
    net.load_state_dict(torch.load(args.weight, map_location=device))
    net.eval()

    all_rows = []
    last_true = None
    last_pred = None
    for i in range(args.test_iter):
        dataset = FewshotDataset(
            test_data,
            test_label,
            episode_num=args.episode_num_test,
            way_num=args.way_num_CWRU,
            shot_num=args.shot_num,
            query_num=args.query_num,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        acc, precision, recall, f1, y_true, y_pred = function.evaluate_fewshot(
            loader, net, device, args.way_num_CWRU, args.shot_num
        )
        all_rows.append({"iter": i + 1, "acc": acc, "precision": precision, "recall": recall, "f1": f1})
        last_true, last_pred = y_true, y_pred
        print(f"Iter {i+1}/{args.test_iter}: acc={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "metrics.csv", index=False)
    summary = df[["acc", "precision", "recall", "f1"]].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary.csv")
    print("\nSummary")
    print(summary)

    if last_true is not None and last_pred is not None:
        save_confusion_matrix(last_true, last_pred, output_dir / "confusion_matrix.png")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
