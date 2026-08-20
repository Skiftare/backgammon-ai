"""Тесты честных костей и стартовой позиции коротких нард."""

from core.dice import RandomDice, expand_double
from core.board import Position


def test_expand_double_regular():
    assert expand_double((5, 2)) == (5, 2)


def test_expand_double_duplicate():
    assert expand_double((3, 3)) == (3, 3, 3, 3)


def test_roll_range():
    d = RandomDice()
    d.seed(42)
    for _ in range(1000):
        a, b = d.roll()
        assert 1 <= a <= 6 and 1 <= b <= 6


def test_roll_deterministic_seed():
    d1, d2 = RandomDice(), RandomDice()
    d1.seed(7); d2.seed(7)
    assert d1.roll_many(20) == d2.roll_many(20)


def test_initial_position_counts():
    pos = Position.initial()
    assert pos.count_white() == 15
    assert pos.count_black() == 15
    assert pos.check_invariants()


def test_starter_layout():
    pos = Position.initial()
    # белые: 24(23):2, 13(12):5, 8(7):3, 6(5):5
    assert pos.at(23) == 2
    assert pos.at(12) == 5
    assert pos.at(7) == 3
    assert pos.at(5) == 5
    # чёрные: зеркально
    assert pos.at(0) == -2
    assert pos.at(11) == -5
    assert pos.at(16) == -3
    assert pos.at(18) == -5