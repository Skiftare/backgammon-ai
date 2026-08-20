"""Контракт хода и обёрточный API движка коротких нард.

В этом модуле:
- `Move` — представление полного легального хода как последовательности
  атомарных шагов `(from_label, to_label)` (формат движка gym-backgammon).
  Метки: `-1 = BAR` (приход с бара), `99 = OFF` (вынос), иначе индекс 0..23.
- Функциональные обёртки делегируют в `core/engine.py` (MIT-движок правил).

Контракт проекта (архитектурный): **модель никогда не генерирует ходы**.
Она выбирает один из `legal_moves(pos, roll)`; среда НЕ может произвести
невозможный ход, т.к. ходы строит детерминированный движок правил.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .dice import DiceRoll

# Метки шага (в нотации движка):
BAR = -1   # приход с бара
OFF = 99   # вынос


@dataclass(frozen=True, slots=True)
class Move:
    steps: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"Move({self.steps})"


# --- обёртки над core/engine (наполняются там, не дублируя логику) ---

def legal_moves(pos, roll: DiceRoll):
    """Все легальные полные ходы для текущего игрока (делегирует в engine)."""
    from .engine import legal_moves as _lm

    return _lm(pos, roll)


def apply_move(pos, move: Move):
    """Применить ход, вернуть новое состояние."""
    from .engine import apply_move as _am

    return _am(pos, move)