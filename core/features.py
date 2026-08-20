"""Представление позиции для нейросети (эмбеддинг состояния).

Цель: стабильный, детерминированный плоский вектор для value/policy сета,
не зависящий от цвета ходящего — т.е. всегда «со стороны текущего игрока»:

- Если ходит белый (в нотации движка белые = `+`, идут к дому 23) — «свои»
  фишки это положительные, «чужие» — отрицательные по модулю.
- Если ходит чёрный — знаки инвертируются, чтобы фишки «текущего игрока»
  снова были «свои» (положительные), а дом/бар его — на месте.

Так сеть не обязана помнить «чей ход» для зеркала: вход всегда в перспективе
ходящего (обычная практика, как в TD-Gammon/AlphaZero self-play).

Кодировка каждой точки: bina-«слои» 1..MAX_ON_POINT (есть ли >=1,>=2,… фишек).
Хвост: бар/дом текущего и соперника + бит очереди (для PC-предсказаний).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import Position, N_POINTS

MAX_ON_POINT = 6


@dataclass(frozen=True, slots=True)
class Encoder:
    max_on_point: int = MAX_ON_POINT
    include_turn: bool = True

    def dim(self) -> int:
        d = 2 * N_POINTS * self.max_on_point
        d += 4  # bar_self, home_self, bar_opp, home_opp
        if self.include_turn:
            d += 1
        return d

    def encode(self, pos: Position) -> np.ndarray:
        if pos.turn == "white":
            self_counts = [max(0, v) for v in pos.points]
            opp_counts = [max(0, -v) for v in pos.points]
            bar_self, home_self = pos.bar_white, pos.home_white
            bar_opp, home_opp = pos.bar_black, pos.home_black
            turn_bit = 1.0
        else:
            # чёрный ход: инвертируем «свои/противник»
            self_counts = [max(0, -v) for v in pos.points]
            opp_counts = [max(0, v) for v in pos.points]
            bar_self, home_self = pos.bar_black, pos.home_black
            bar_opp, home_opp = pos.bar_white, pos.home_white
            turn_bit = 0.0

        own = self._encode_side(self_counts, self.max_on_point)
        opp = self._encode_side(opp_counts, self.max_on_point)
        tail = np.array(
            [bar_self / 15.0, home_self / 15.0, bar_opp / 15.0, home_opp / 15.0],
            dtype=float,
        )
        if self.include_turn:
            tail = np.concatenate([tail, np.array([turn_bit], dtype=float)])
        return np.concatenate([own, opp, tail])

    @staticmethod
    def _encode_side(counts: list[int], max_on_point: int) -> np.ndarray:
        out = []
        for cnt in counts:
            capped = min(cnt, max_on_point)
            out.extend(1.0 if capped >= k else 0.0 for k in range(1, max_on_point + 1))
        return np.array(out, dtype=float)

    def __repr__(self) -> str:
        return f"Encoder(dim={self.dim()})"