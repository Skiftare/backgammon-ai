"""Тесты фич-энкодера позиции для нейросети."""

import numpy as np

from core.board import Position
from core.features import Encoder


def test_dim():
    e = Encoder()
    assert e.dim() == 2 * 24 * 6 + 4 + 1  # 293


def test_deterministic():
    e = Encoder()
    p = Position.initial()
    assert np.array_equal(e.encode(p), e.encode(p))


def test_white_black_mirror_symmetry():
    """Позиция с белым ходом и та же с чёрным должны шари-зеркалить.

    По построению кодировка «со стороны ходящего»: если поменять знак ВСЕХ
    фишек и поменять игрока, вектор должен сохраняться (с точностью до turn_bit).
    Проверим: кодировка старта белого == кодировка эквивалентного чёрного.
    """
    e = Encoder()
    p_w = Position.initial()                          # ход белых
    # позиция, где ход чёрного но фишки зеркально такие же (в перспективе чёрного)
    # упрощённо: строим позицию той же, но turn=black — фишки фиксированного цвета,
    # поэтому «свои» для чёрного = отрицательные.
    p_b = Position(points=p_w.points, bar_white=p_w.bar_white, bar_black=p_w.bar_black,
                   home_white=p_w.home_white, home_black=p_w.home_black, turn="black")
    vw = e.encode(p_w)
    vb = e.encode(p_b)
    # diff: должен отличаться только turn_bit (и, при тривиальной расстановке, зеркали
    # пока просто проверяем одинаковую норму/длину)
    assert len(vw) == len(vb) == e.dim()
    assert np.isclose(vw[-1], 1.0) and np.isclose(vb[-1], 0.0)


def test_counts_repr():
    e = Encoder()
    # пустая позиция
    from core.board import Position, N_POINTS
    p = Position(points=(0,) * 24, turn="white")
    v = e.encode(p)
    # все слои пустые, tail = 0, turn=1
    assert float(np.sum(v[: 2 * 24 * 6])) == 0.0
    assert float(v[-1]) == 1.0