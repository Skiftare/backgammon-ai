"""Честные кости для нардов: детерминированно-сидируемые, независимые броски.

Среда должна быть детерминированной настолько, насколько это позволяет
«полное доверие»: пользователю можно дать самому кидать кости и вводить
значения через интерфейс. Генератор здесь — просто источник равномерных
целых, а не часть «мозга» модели.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

DIE_MIN = 1
DIE_MAX = 6

# Пара значений двух кубиков.
DoubledDice = tuple[int, int]
# Развёрнутая в последовательность доступных «шагов» (дубль → 4 одинаковых).
DiceRoll = tuple[int, ...]


def expand_double(roll: DoubledDice) -> DiceRoll:
    """Развернуть два кубика в кортеж доступныхшагов по правилам.

    Дубль (a==a) → четыре одинаковых шага; не-дубль → (a, b).
    Например: (3,3) → (3,3,3,3); (5,2) → (5,2).
    """
    a, b = roll
    if a == b:
        return (a, a, a, a)
    return (a, b)


@dataclass(slots=True)
class RandomDice:
    """Честный бросок двух кубиков с собственной детерминированной RNG.

    `seed()` задаёт воспроизводимость. Экземпляр изолирован от внешнего
    `random`-состояния (не ломает остальное приложение).
    """

    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def seed(self, value: int | None = None) -> None:
        self._rng.seed(value)

    def roll(self) -> DoubledDice:
        """Бросок двух честных, независимых кубиков."""
        a = self._rng.randint(DIE_MIN, DIE_MAX)
        b = self._rng.randint(DIE_MIN, DIE_MAX)
        return (a, b)

    def roll_many(self, n: int) -> list[DoubledDice]:
        """n бросков подряд (для тестов/replay)."""
        return [self.roll() for _ in range(n)]


def iter_shuffled_rolls(rolls: Sequence[DoubledDice], *, seed: int | None = None) -> Iterator[DoubledDice]:
    """Детерминированное перемешивание списка бросков (для replay экспериментов)."""
    rng = random.Random(seed)
    idx = list(range(len(rolls)))
    rng.shuffle(idx)
    for i in idx:
        yield rolls[i]