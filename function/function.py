"""Utility functions for SC-MambaFew input-representation experiments.

This file extends the original SC-MambaFew utilities with six signal-to-image
representations used in the input representation analysis paper:
SP, SC, GS, RP, MTF, and GAF.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import cv2
import librosa
import numpy as np
import pywt
import scipy.spatial.distance as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_func(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Signal-to-image transforms
# All functions receive shape (N, L) numpy arrays and return torch tensors
# with shape (N, 1, res, res).
# -----------------------------------------------------------------------------
def _to_tensor_image(images: List[np.ndarray]) -> torch.Tensor:
    arr = np.stack(images).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(1)


def _minmax_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    mn, mx = float(np.min(signal)), float(np.max(signal))
    return (signal - mn) / (mx - mn + eps)


def to_spectrum(data: np.ndarray, res: int = 64) -> torch.Tensor:
    """STFT spectrogram (SP)."""
    images: List[np.ndarray] = []
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, :], dtype=np.float32)
        spectrogram = librosa.stft(signal, n_fft=512, hop_length=512)
        power = np.abs(spectrogram) ** 2
        log_spectrogram = librosa.power_to_db(power)
        log_spectrogram = cv2.resize(log_spectrogram, (res, res), interpolation=cv2.INTER_AREA)
        images.append(log_spectrogram)
    return _to_tensor_image(images)


def to_SC(data: np.ndarray, wavelet: str = "morl", res: int = 64) -> torch.Tensor:
    """Continuous wavelet scalogram (SC)."""
    images: List[np.ndarray] = []
    scales = np.arange(1, 128)
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, :], dtype=np.float32)
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
        scalogram = np.abs(coeffs)
        scalogram = cv2.resize(scalogram, (res, res), interpolation=cv2.INTER_AREA)
        images.append(scalogram)
    return _to_tensor_image(images)


def to_GS(data: np.ndarray, res: int = 64) -> torch.Tensor:
    """Gray-scale direct encoding (GS)."""
    images: List[np.ndarray] = []
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, :], dtype=np.float32)
        side = int(np.ceil(np.sqrt(len(signal))))
        padded = np.pad(signal, (0, side * side - len(signal)), mode="constant")
        image = padded.reshape(side, side)
        max_abs = np.max(np.abs(image))
        if max_abs > 0:
            image = image / max_abs
        image = cv2.resize(image, (res, res), interpolation=cv2.INTER_AREA)
        images.append(image)
    return _to_tensor_image(images)


def to_GAF(data: np.ndarray, res: int = 64) -> torch.Tensor:
    """Gramian Angular Summation Field (GAF/GASF)."""
    images: List[np.ndarray] = []
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, :], dtype=np.float32)
        # Scale to [-1, 1].
        x = 2.0 * _minmax_normalize(signal) - 1.0
        x = np.clip(x, -1.0, 1.0)
        phi = np.arccos(x)
        gasf = np.cos(phi[:, None] + phi[None, :])
        gasf = cv2.resize(gasf, (res, res), interpolation=cv2.INTER_AREA)
        images.append(gasf)
    return _to_tensor_image(images)


def to_MTF(data: np.ndarray, num_bins: int = 10, res: int = 64) -> torch.Tensor:
    """Markov Transition Field (MTF).

    This implementation follows the original experimental code: it computes an
    L x L transition field and then resizes it to the requested image size.
    It can be slow for long signals; keep this in mind when running all methods.
    """
    images: List[np.ndarray] = []
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, :], dtype=np.float32)
        mn, mx = float(np.min(signal)), float(np.max(signal))
        if mn == mx:
            discretized = np.zeros_like(signal, dtype=np.int64)
        else:
            bins = np.linspace(mn, mx, num_bins + 1)
            discretized = np.digitize(signal, bins) - 1
            discretized = np.clip(discretized, 0, num_bins - 1).astype(np.int64)

        transition = np.zeros((num_bins, num_bins), dtype=np.float32)
        for j in range(len(discretized) - 1):
            transition[discretized[j], discretized[j + 1]] += 1.0
        row_sums = transition.sum(axis=1, keepdims=True)
        transition = np.divide(transition, row_sums, out=np.zeros_like(transition), where=row_sums != 0)

        mtf = transition[discretized[:, None], discretized[None, :]]
        mtf = cv2.resize(mtf, (res, res), interpolation=cv2.INTER_AREA)
        images.append(mtf)
    return _to_tensor_image(images)


def to_RP(data: np.ndarray, threshold: float = 0.1, res: int = 64) -> torch.Tensor:
    """Recurrence Plot (RP)."""
    images: List[np.ndarray] = []
    for i in range(data.shape[0]):
        signal = np.asarray(data[i, :], dtype=np.float32)
        signal_2d = signal.reshape(-1, 1)
        distance_matrix = dist.cdist(signal_2d, signal_2d, metric="euclidean")
        rp = np.where(distance_matrix < threshold, 1.0, 0.0).astype(np.float32)
        rp = cv2.resize(rp, (res, res), interpolation=cv2.INTER_AREA)
        images.append(rp)
    return _to_tensor_image(images)


TRANSFORMS: Dict[str, Callable[..., torch.Tensor]] = {
    "SP": to_spectrum,
    "SC": to_SC,
    "GS": to_GS,
    "RP": to_RP,
    "MTF": to_MTF,
    "GAF": to_GAF,
}


def get_transform(name: str) -> Callable[..., torch.Tensor]:
    name = name.upper()
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform '{name}'. Choose from {list(TRANSFORMS)}")
    return TRANSFORMS[name]


# -----------------------------------------------------------------------------
# Few-shot helpers
# -----------------------------------------------------------------------------
def group_support_by_class(
    support_images: torch.Tensor,
    support_targets: torch.Tensor,
    way_num: int,
    shot_num: int,
    device: torch.device | str,
) -> List[torch.Tensor]:
    """Convert batched support tensors into class-wise support tensors.

    DataLoader output shape is usually:
        support_images:  (B=1, way*shot, C, H, W)
        support_targets: (B=1, way*shot)
    The model expects a list with length=way_num; each item has shape
    (shot_num, C, H, W).
    """
    if support_images.dim() == 5:
        support_images = support_images.squeeze(0)
    if support_targets.dim() == 2:
        support_targets = support_targets.squeeze(0)

    grouped: List[torch.Tensor] = []
    for cls in range(way_num):
        idx = torch.nonzero(support_targets == cls, as_tuple=False).flatten()[:shot_num]
        if idx.numel() < shot_num:
            raise RuntimeError(
                f"Not enough support samples for class {cls}: "
                f"found {idx.numel()}, required {shot_num}."
            )
        grouped.append(support_images[idx].to(device))
    return grouped


class ContrastiveLoss(nn.Module):
    """Cross-entropy implementation of the score-based few-shot objective."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(output, target)


@torch.no_grad()
def evaluate_fewshot(loader, net, device, way_num: int, shot_num: int) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    net.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.squeeze(0).to(device)  # (way*query, C, H, W)
        targets = query_targets.squeeze(0).long().to(device)
        support_grouped = group_support_by_class(support_images, support_targets, way_num, shot_num, device)

        for i in range(q.size(0)):
            scores, _, _ = net(q[i:i + 1], support_grouped)
            pred = int(torch.argmax(scores.float(), dim=-1).item())
            y_pred.append(pred)
            y_true.append(int(targets[i].item()))

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    return float(acc), float(precision), float(recall), float(f1), y_true_arr, y_pred_arr


def add_gaussian_noise_snr(data: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to match a target SNR in dB."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(data ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=data.shape)
    return (data + noise).astype(np.float32)
