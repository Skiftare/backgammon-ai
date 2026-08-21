"""Нейронная сеть для нард: MLP value-сеть (архитектура в стиле TD-Gammon).

Классика: вход — фич-вектор позиции (293-d), скрытый слой(и) с tanh, выход —
одно число (value, ожидание выигрыша из позиции для ходящего). Плюс (на будущее)
policy-голова для выбора среди легальных ходов.

Здесь — **value-only** (как у Tesauro): выбор хода делается greedy по value
(применяем ход, смотрим value результирующей позиции). Это чистый классический
путь. Policy-голову добавим позже, если захотим ускорить self-play.

Обучается стандартным BP (backprop) как исходник; contrastive-PC/EP — потом.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> (B, 1) value в [-1,1] (Tanh на выходе как в td-gammon)
        return torch.tanh(self.net(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Удобная обёртка: x -> скаляр value."""
        return self.forward(x).squeeze(-1)


def make_value_net(in_dim: int, hidden: int = 128, seed: int | None = None) -> ValueNet:
    if seed is not None:
        torch.manual_seed(seed)
    return ValueNet(in_dim, hidden)