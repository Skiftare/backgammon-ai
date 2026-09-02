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
    def __init__(self, in_dim: int, hidden: int = 512, out: int = 1, layers: int = 2) -> None:
        super().__init__()
        mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(max(1, layers) - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, out)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> (B, 1) value в [-1,1] (Tanh на выходе как в td-gammon)
        return torch.tanh(self.net(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Удобная обёртка: x -> скаляр value."""
        return self.forward(x).squeeze(-1)


def make_value_net(in_dim: int, hidden: int = 512, layers: int = 2,
                   seed: int | None = None) -> ValueNet:
    if seed is not None:
        torch.manual_seed(seed)
    return ValueNet(in_dim, hidden=hidden, layers=layers)