from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import function.function as function
from CWRU.CWRU_dataset import CWRU
from dataloader.dataloader import FewshotDataset
from net.new_proposed import MainNet


CLASS_NAMES = [
    "Normal",
    "0.007-Ball",
    "0.014-Ball",
    "0.021-Ball",
    "0.007-InnerRace",
    "0.014-InnerRace",
    "0.021-InnerRace",
    "0.007-OuterRace6",
    "0.014-OuterRace6",
    "0.021-OuterRace6",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SC-MambaFew input representation training on CWRU")
    parser.add_argument("--dataset", default="CWRU", choices=["CWRU"])
    parser.add_argument("--data_path", default="./CWRU", help="Path where CWRU .mat files are stored/downloaded")
    parser.add_argument("--transform", default="SP", choices=["SP", "SC", "GS", "RP", "MTF", "GAF"])
    parser.add_argument("--resolution", type=int, default=64, choices=[64, 128], help="Input image resolution")
    parser.add_argument("--shot_num", type=int, default=1, choices=[1, 5])
    parser.add_argument("--training_samples_CWRU", type=int, default=None, help="Default: 30 for 1-shot, 150 for 5-shot")
    parser.add_argument("--episode_num_train", type=int, default=100)
    parser.add_argument("--episode_num_test", type=int, default=75)
    parser.add_argument("--way_num_CWRU", type=int, default=10)
    parser.add_argument("--query_num", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    return parser.parse_args()


def prepare_cwru(args):
    window_size = 2048
    if args.training_samples_CWRU is None:
        args.training_samples_CWRU = 30 if args.shot_num == 1 else 150
    if args.training_samples_CWRU % 30 != 0:
        raise ValueError("training_samples_CWRU must be a multiple of 30 for the selected CWRU setup.")

    split = args.training_samples_CWRU // 30
    data = CWRU(split, ["12DriveEndFault"], ["1772", "1750", "1730"], window_size, args.data_path)

    # 2048 time points x 2 channels (DE + FE) -> 4096 vector.
    train_x = data.X_train.astype(np.float32).reshape([args.training_samples_CWRU, 4096])
    test_x = data.X_test.astype(np.float32).reshape([750, 4096])

    transform_fn = function.get_transform(args.transform)
    train_data = transform_fn(train_x, res=args.resolution)
    test_data = transform_fn(test_x, res=args.resolution)

    train_label = torch.from_numpy(data.y_train).long()
    test_label = torch.from_numpy(data.y_test).long()
    return train_data, train_label, test_data, test_label


def set_batchnorm_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def make_loaders(args, train_data, train_label, test_data, test_label):
    train_dataset = FewshotDataset(
        train_data,
        train_label,
        episode_num=args.episode_num_train,
        way_num=args.way_num_CWRU,
        shot_num=args.shot_num,
        query_num=args.query_num,
    )
    test_dataset = FewshotDataset(
        test_data,
        test_label,
        episode_num=args.episode_num_test,
        way_num=args.way_num_CWRU,
        shot_num=args.shot_num,
        query_num=args.query_num,
    )
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0),
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
    )


def train_one_epoch(net, loader, criterion, optimizer, device, args):
    net.train()
    set_batchnorm_eval(net)
    running_loss = 0.0

    for query_images, query_targets, support_images, support_targets in tqdm(loader, leave=False):
        q = query_images.squeeze(0).to(device)
        targets = query_targets.squeeze(0).long().to(device)
        support_grouped = function.group_support_by_class(
            support_images, support_targets, args.way_num_CWRU, args.shot_num, device
        )

        optimizer.zero_grad()
        total_loss = 0.0
        for i in range(q.size(0)):
            out, _, _ = net(q[i:i + 1], support_grouped)
            loss = criterion(out, targets[i:i + 1])
            total_loss = total_loss + loss

        avg_loss = total_loss / q.size(0)
        avg_loss.backward()
        optimizer.step()
        running_loss += float(avg_loss.item())

    return running_loss / max(len(loader), 1)


def main():
    args = parse_args()
    function.seed_func(args.seed)
    device = torch.device(args.device)

    run_name = f"cwru_{args.shot_num}shot_{args.resolution}_{args.transform}"
    output_dir = Path(args.output_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data, train_label, test_data, test_label = prepare_cwru(args)
    train_loader, test_loader = make_loaders(args, train_data, train_label, test_data, test_label)

    h = args.resolution // 4
    net = MainNet(h=h, w=h).to(device)
    criterion = function.ContrastiveLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_acc = -1.0
    history = []
    best_path = checkpoint_dir / "best.pth"

    print(f"Training {run_name}")
    print(f"Train data: {tuple(train_data.shape)} | Test data: {tuple(test_data.shape)}")

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device, args)
        scheduler.step()

        acc, precision, recall, f1, _, _ = function.evaluate_fewshot(
            test_loader, net, device, args.way_num_CWRU, args.shot_num
        )
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.num_epochs} | "
            f"loss={train_loss:.4f} acc={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), best_path)
            print(f"  Saved best checkpoint: {best_path} (acc={best_acc:.4f})")

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"Done. Best acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
