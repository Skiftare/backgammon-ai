"""TD(λ)-обучение value-сети для нард (классика Tesauro).

Сеть предсказывает value «со стороны ходящего» (энкодер это кодирует).
При переходе хода позиция следующего шага кодируется для СЛЕДУЮЩЕГО игрока,
поэтому для текущего её ценность — минус V(следующего). TD-error:

    delta_t = r_t + gamma*(-V(s_{t+1})) - V(s_t)

где r_t == 0 для всех шагов, кроме последнего, где r = +1 (победил ходящий)
или -1 (победил противник).

Всё считается тензорно (без отрыва от графа) — backward идёт по параметрам
сети. Это BP-baseline; PC/EP — следующий этап.
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
    gamma: float = 1.0,          # no discount; до конца
    lr: float = 1e-3,
) -> float:
    """Один градиентный шаг по траектории self-play.

    Возвращает TD-loss (средний квадрат дельт) для мониторинга.
    """
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer.zero_grad()

    Xs = torch.stack([torch.tensor(encoder.encode(p), dtype=torch.float32) for p in traj])
    V = net.value(Xs)            # (T,) value «для ходящего» каждой позиции (с графом)

    T = len(traj)
    r_last = SIGNS[winner]      # +1/−1
    r_t = torch.zeros(T, dtype=torch.float32)
    r_t[-1] = r_last

    # обратный проход: delta_t = r_t + gamma*(-V_{t+1}) - V_t
    # (V_{t+1} — value СЛЕДУЮЩЕГО игрока; для текущего → -V_{t+1})
    deltas = torch.zeros(T, dtype=torch.float32)
    v_next = torch.zeros((), dtype=torch.float32)
    for t in range(T - 1, -1, -1):
        delta = r_t[t] + gamma * (-v_next) - V[t]
        deltas[t] = delta
        v_next = V[t]           # тензор, участвует в след. дельте (граф цел)

    loss = (deltas ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())