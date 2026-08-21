"""Честные броски кубиков (обёртка для экспериментов/веба)."""

from __future__ import annotations

import random

_rng = random.Random()


def roll() -> int:
    """Значение одной честной игральной кости (1..6)."""
    return _rng.randint(1, 6)


def roll_dice() -> tuple[int, int]:
    """Две кости."""
    return (roll(), roll())


def seed(v: int | None = None) -> None:
    _rng.seed(v)