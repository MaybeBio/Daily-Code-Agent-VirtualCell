from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def pearson(prediction: Tensor, target: Tensor) -> Tensor:
    left = prediction.flatten() - prediction.mean()
    right = target.flatten() - target.mean()
    return (left * right).sum() / (left.square().sum().sqrt() * right.square().sum().sqrt()).clamp_min(1e-12)


def binary_auc(score: Tensor, target: Tensor) -> Tensor:
    order = score.argsort()
    rank = torch.empty_like(order, dtype=torch.float64)
    rank[order] = torch.arange(1, len(score) + 1, dtype=torch.float64, device=score.device)
    positive = target.bool()
    count_positive = positive.sum()
    count_negative = (~positive).sum()
    numerator = rank[positive].sum() - count_positive * (count_positive + 1) / 2
    return numerator / (count_positive * count_negative).clamp_min(1)


def macro_f1(prediction: Tensor, target: Tensor, classes: int) -> Tensor:
    scores = []
    for label in range(classes):
        true_positive = ((prediction == label) & (target == label)).sum()
        false_positive = ((prediction == label) & (target != label)).sum()
        false_negative = ((prediction != label) & (target == label)).sum()
        scores.append(2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1))
    return torch.stack(scores).float().mean()


def dynamic_time_warping(left: Tensor, right: Tensor) -> Tensor:
    rows, columns = left.shape[0], right.shape[0]
    table = torch.full((rows + 1, columns + 1), float("inf"), device=left.device)
    table[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            cost = torch.linalg.vector_norm(left[row - 1] - right[column - 1])
            table[row, column] = cost + torch.stack((table[row - 1, column], table[row, column - 1], table[row - 1, column - 1])).min()
    return table[rows, columns]


def cross_boundary_direction(velocity: Tensor, source: Tensor, destination: Tensor) -> Tensor:
    direction = destination - source
    numerator = (velocity * direction).sum(dim=-1)
    denominator = torch.linalg.vector_norm(velocity, dim=-1) * torch.linalg.vector_norm(direction, dim=-1)
    return (numerator / denominator.clamp_min(1e-12)).mean()


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 10000, level: float = 0.95) -> Interval:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    statistics = values[indices].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return Interval(float(values.mean()), float(np.quantile(statistics, alpha)), float(np.quantile(statistics, 1.0 - alpha)))
