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

from collections.abc import Sequence

import torch

from core.features import Encoder
from model.net import ValueNet

SIGNS = {"white": 1.0, "black": -1.0}


class ValueTrainer:
    """Держит сеть, оптимизатор и GradScaler между шагами.

    - ВЕСЬ батч партий кодируется в один большой тензор X и прогоняется ОДНИМ
      forward/backward — это максимально эффективная загрузка GPU.
    - fp16 (autocast + GradScaler) на CUDA ускоряет matmul на 3050 в ~2 раза.
    """

    def __init__(self, net: ValueNet, lr: float = 1e-3, device: str = "cpu",
                 fp16: bool = True) -> None:
        self.net = net.to(device)
        self.device = device
        self.opt = torch.optim.SGD(net.parameters(), lr=lr)
        self.fp16 = bool(fp16 and device == "cuda" and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

    def step(self, games, encoder: Encoder, lam: float = 0.7, gamma: float = 1.0) -> float:
        net = self.net
        # 1) кодируем ВСЕ позиции всех партий ОДНИМ батч-энкодером (векторно)
        all_pos: list = []
        spans: list[tuple[int, int, str]] = []
        for traj, winner in games:
            s = len(all_pos)
            all_pos.extend(traj)
            spans.append((s, len(all_pos), winner))
        X = torch.tensor(encoder.encode_batch(all_pos), dtype=torch.float32, device=self.device)

        self.opt.zero_grad()
        use = torch.autocast(device_type="cuda", enabled=self.fp16)
        with use:
            V = net.value(X)  # один forward на весь батч
            total = torch.zeros((), dtype=torch.float32, device=self.device)
            for (s, e, winner) in spans:
                Vg = V[s:e]
                T = e - s
                # delta_t = r_t + gamma*(-V_{t+1}) - V_t ; для t=T-1: r=sign
                Vshift = torch.cat([Vg[1:T], torch.zeros(1, dtype=Vg.dtype, device=Vg.device)])
                r = torch.zeros_like(Vg)
                r[-1] = SIGNS[winner]
                delta = r + gamma * (-Vshift) - Vg
                total = total + delta.pow(2).mean()
            loss = total / max(1, len(games))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.item())