"""Self-play: генерация партии с greedy-политикой по value-сети.

Обучение value-сети (TD) идёт по траекториям, которые генерят эти партии.
Каждый ход: бросок костей (честный), список легальных ходов из движка, выбор
жадным по value позиции после хода.

Победа: у игрока вынесены все 15 фишек (в нашей нотации home_white/home_black == 15).
Если ходов нет (ход пропадает) — просто смена стороны без передвижений.
"""

from __future__ import annotations

from typing import Optional

import random

import torch

from core.board import Position, TOTAL_CHECKERS
from core.game import legal_moves, apply_move
from core.features import Encoder
from model.net import ValueNet


def _other(turn: str) -> str:
    return "black" if turn == "white" else "white"


def choose_greedy(pos: Position, roll, net: ValueNet, encoder: Encoder) -> Position:
    """Применить лучший ход greedy по value (для текущего игрока)."""
    moves = legal_moves(pos, roll)
    if not moves:
        # хода нет — смена стороны (фишки не трогаем)
        return Position(
            points=pos.points,
            bar_white=pos.bar_white,
            bar_black=pos.bar_black,
            home_white=pos.home_white,
            home_black=pos.home_black,
            turn=_other(pos.turn),
        )

    best: Optional[Position] = None
    best_v: Optional[float] = None
    for m in moves:
        nxt = apply_move(pos, m)
        x = torch.tensor(encoder.encode(nxt), dtype=torch.float32).unsqueeze(0)
        v = float(net.value(x).detach())
        # value оценивается «со стороны ходящего следующего»; текущий игрок хочет
        # позицию, выгодную себе, т.е. для белого — максимум value (после хода ходит
        # чёрный, но value всегда «со стороны ходящего», так НЕ знак не ставим —
        # эвристика: белый себе хочет большой value следующей позиции; чёрный — малый).
        if pos.turn == "white":
            if best is None or v > best_v:
                best, best_v = nxt, v
        else:
            if best is None or v < best_v:
                best, best_v = nxt, v
    return best if best is not None else pos


def play_one_game(net: ValueNet, encoder: Encoder, rng=None, max_steps: int = 1000):
    """Играет одну партию greedy-политикой.

    Возвращает (траектория позиций, победитель_цвет) где победитель = 'white'|'black'.
    На исчерпание max_steps возвращается лидер по числу вынесенных (безопасность).
    """
    if rng is None:
        rng = random.Random()
    pos = Position.initial()
    traj = [pos]
    for _ in range(max_steps):
        # победа?
        if pos.home_white == TOTAL_CHECKERS:
            return traj, "white"
        if pos.home_black == TOTAL_CHECKERS:
            return traj, "black"

        a = rng.randint(1, 6)
        b = rng.randint(1, 6)
        roll = (a, a, a, a) if a == b else (a, b)

        nxt = choose_greedy(pos, roll, net, encoder)
        traj.append(nxt)
        pos = nxt

    winner = "white" if pos.home_white >= pos.home_black else "black"
    return traj, winner