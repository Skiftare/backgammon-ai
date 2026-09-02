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

from .board import N_POINTS, Position

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
        return self.encode_batch([pos])[0]

    def encode_batch(self, positions: list[Position]) -> np.ndarray:
        """Векторно закодировать список позиций в один np-массив (N, dim).

        Весь энкодинг считаем numpy-операциями на массиве (N,24) — это снимает
        per-position Python-цикл и позволяет отдать GPU ОДИН большой тензор.
        """
        N = len(positions)
        P = np.zeros((N, N_POINTS), dtype=float)
        for i, p in enumerate(positions):
            P[i] = p.points
        # маска: белый ход => свои = +points, чёрный => свои = -points
        turn_w = np.array([p.turn == "white" for p in positions], dtype=bool)
        M = self.max_on_point

        values = np.where(turn_w[:, None], P, -P)      # (N,24) свои (полож.)
        own_counts = np.maximum(values, 0)
        opp_counts = np.maximum(-values, 0)
        own = self._encode_side_batch(own_counts, M)   # (N, 24*M)
        opp = self._encode_side_batch(opp_counts, M)

        bars_self = np.where(turn_w, np.array([p.bar_white for p in positions], dtype=float),
                             np.array([p.bar_black for p in positions], dtype=float))
        homes_self = np.where(turn_w, np.array([p.home_white for p in positions], dtype=float),
                              np.array([p.home_black for p in positions], dtype=float))
        bars_opp = np.where(turn_w, np.array([p.bar_black for p in positions], dtype=float),
                            np.array([p.bar_white for p in positions], dtype=float))
        homes_opp = np.where(turn_w, np.array([p.home_black for p in positions], dtype=float),
                             np.array([p.home_white for p in positions], dtype=float))

        tail_cols = [bars_self / 15.0, homes_self / 15.0, bars_opp / 15.0, homes_opp / 15.0]
        tail = np.stack(tail_cols, axis=1)
        if self.include_turn:
            tail = np.concatenate([tail, turn_w[:, None].astype(float)], axis=1)

        return np.concatenate([own, opp, tail], axis=1)

    @staticmethod
    def _encode_side_batch(counts: np.ndarray, max_on_point: int) -> np.ndarray:
        counts = np.minimum(counts, max_on_point)
        # (N,24, M): counts[..., None] >= arange(M) -> пороговая маска
        mask = counts[:, :, None] >= np.arange(1, max_on_point + 1)[None, None, :]
        return mask.reshape(counts.shape[0], -1).astype(float)

    @staticmethod
    def _encode_side(counts: list[int], max_on_point: int) -> np.ndarray:
        arr = np.minimum(np.asarray(counts, dtype=float), max_on_point)
        # (M,24): для каждой точки — маска >=1, >=2, ... >=M (# фишек)
        mask = arr[None, :] >= np.arange(1, max_on_point + 1)[:, None]
        return mask.ravel().astype(float)

    def __repr__(self) -> str:
        return f"Encoder(dim={self.dim()})"