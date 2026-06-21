from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FewshotDataset(Dataset):
    """Episode sampler for N-way K-shot classification.

    This dataset intentionally keeps tensors on CPU. Move them to CUDA only in
    the train/test loop. This makes the code portable and avoids CUDA work inside
    DataLoader workers.
    """

    def __init__(self, train_data, train_label, episode_num=1000, way_num=10, shot_num=1, query_num=1):
        self.train_data = train_data
        self.train_label = train_label.long() if torch.is_tensor(train_label) else torch.as_tensor(train_label).long()
        self.episode_num = int(episode_num)
        self.way_num = int(way_num)
        self.shot_num = int(shot_num)
        self.query_num = int(query_num)

    def __len__(self):
        return self.episode_num

    def __getitem__(self, index):
        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        perm = torch.randperm(len(self.train_label))
        labels_perm = self.train_label[perm]

        for label_num in range(self.way_num):
            cls_positions = torch.nonzero(labels_perm == label_num, as_tuple=False).flatten()
            required = self.shot_num + self.query_num
            if cls_positions.numel() < required:
                raise RuntimeError(
                    f"Class {label_num} has {cls_positions.numel()} samples in this episode, "
                    f"but {required} are required. Increase training samples or reduce shot/query."
                )

            selected = cls_positions[:required]
            support_pos = selected[:self.shot_num]
            query_pos = selected[self.shot_num:self.shot_num + self.query_num]

            support_idx = perm[support_pos]
            query_idx = perm[query_pos]

            support_images.append(self.train_data[support_idx])
            support_targets.append(torch.full((self.shot_num,), label_num, dtype=torch.long))
            query_images.append(self.train_data[query_idx])
            query_targets.append(torch.full((self.query_num,), label_num, dtype=torch.long))

        support_images = torch.cat(support_images, dim=0)
        support_targets = torch.cat(support_targets, dim=0)
        query_images = torch.cat(query_images, dim=0)
        query_targets = torch.cat(query_targets, dim=0)

        return query_images, query_targets, support_images, support_targets
