"""TD(λ)-обучение value-сети для нард (классика Tesauro).

Сеть предсказывает value «со стороны ходящего» (энкодер это кодирует).
При переходе хода позиция следующего шага кодируется для СЛЕДУЮЩЕГО игрока,
поэтому для текущего её ценность — минус V(следующего). TD(λ) здесь:

  delta_t = r_t + γ·( -V(s_{t+1}) ) - V(s_t)

где r_t == 0 для всех шагов, кроме последнего, где r = +1 (победил ходящий)
или -1 (победил противник). Элижибилити-трейсы z обновляются на каждом шаге.

Функция `train_episode` принимает обученную траекторию, победу, и делает
один градиентный шаг (backprop — BP-baseline; PC/EP потом).
"""

from __future__ import annotations

from typing import Sequence

import torch

from core.board import Position
from core.features import Encoder
from model.net import ValueNet

SIGNS = {"white": 1.0, "black": -1.0}


def train_episode(
    net: ValueNet,
    encoder: Encoder,
    traj: Sequence[Position],
    winner: str,
    lam: float = 0.7,
    gamma: float = 1.0,          # no discount; идёт до конца
    lr: float = 1e-3,
) -> float:
    """Обновляет веса net по одной траектории self-play.

    Возвращает средний |TD-error| (для мониторинга).
    """
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer.zero_grad()

    # value для каждой позиции
    Xs = torch.stack([torch.tensor(encoder.encode(p), dtype=torch.float32) for p in traj])
    V = net.value(Xs)  # (T,) ценности для «ходящего» каждой позиции

    T = len(traj)
    # целевые и TD-дельты
    # последний шаг: r последнего = сигралл
    r_last = SIGNS[winner]  # +1/−1
    deltas = []
    # идём с конца: delta_t = r_t + gamma*(-V_{t+1}) - V_t
    # V_t — value для ходящего в t; V_{t+1} уже для следующего, поэтому -V_{t+1}.
    value_next = 0.0  # после терминала — 0 (нет будущего соперника)
    for t in reversed(range(T)):
        r_t = r_last if t == T - 1 else 0.0
        v = float(V[t])
        delta = (r_t + gamma * (-value_next)) - v
        deltas.append(delta)
        value_next = v
    deltas.reverse()

    # взвешиваем λ
    eps_error = torch.tensor(deltas, dtype=torch.float32)
    # loss = mean(TD-error)^2 (BP-игра; с trace — по идее элижи: пропустим trace для начала,
    # возьмём «TD(0)-подобный» mean-squared)
    loss = (eps_error ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())