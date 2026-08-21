"""Позиция коротких нард (нотация, согласованная с MIT-движком gym-backgammon).

Нотация (ровно как в движке, `core/_engine/backgammon.py:init_board`):

- `points`: 24 элемента (индекс 0..23). `+n` — n белых фишек, `-n` — n чёрных,
  0 — пусто.
- **Белые** (WHITE=0 в движке) двигаются **вверх** (к индексу 23); их **дом** —
  индексы 18..23. Старт: белые на `{5:5, 7:3, 12:5, 23:2}`.
- **Чёрные** (BLACK=1) двигаются **вниз** (к индексу 0); их **дом** — индексы
  0..5. Старт: чёрные на `{0:-2? нет, 0: -2}` — точнее: `{0: -2, 11: -5, 16: -3,
  18: -5}`.
- `bar_white` / `bar_black` — сбитые (бар), `home_white` / `home_black` — вынесенные.
- `turn` — чей ход.

Точное соответствие движку — гарантируется тестом против `init_board()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

N_POINTS = 24
TOTAL_CHECKERS = 15


@dataclass(frozen=True, slots=True)
class Position:
    points: tuple[int, ...] = field(default_factory=lambda: (0,) * N_POINTS)
    bar_white: int = 0
    bar_black: int = 0
    home_white: int = 0
    home_black: int = 0
    turn: str = "white"  # 'white' | 'black'

    def __post_init__(self) -> None:
        if len(self.points) != N_POINTS:
            raise ValueError(f"ожидалось {N_POINTS} пунктов, получено {len(self.points)}")
        if self.turn not in ("white", "black"):
            raise ValueError(f"неизвестный игрок: {self.turn!r}")

    @classmethod
    def initial(cls) -> Position:
        """Стартовая расстановка коротких нард (совпадает с движком)."""
        pts = [0] * N_POINTS
        # белые (положительные) — как в init_board(): board[5], [7], [12], [23]
        pts[5] = 5
        pts[7] = 3
        pts[12] = 5
        pts[23] = 2
        # чёрные (отрицательные) — init_board(): board[0], [11], [16], [18]
        pts[0] = -2
        pts[11] = -5
        pts[16] = -3
        pts[18] = -5
        return cls(points=tuple(pts), turn="white")

    # ---------- вспомогательное ----------

    def at(self, point: int) -> int:
        return self.points[point]

    def count_white(self) -> int:
        return sum(max(0, p) for p in self.points) + self.bar_white + self.home_white

    def count_black(self) -> int:
        return sum(max(0, -p) for p in self.points) + self.bar_black + self.home_black

    def check_invariants(self) -> bool:
        return self.count_white() == TOTAL_CHECKERS and self.count_black() == TOTAL_CHECKERS

    def iter_points(self) -> list[tuple[int, int]]:
        return [(i, v) for i, v in enumerate(self.points) if v]

    def __repr__(self) -> str:
        return (
            f"Position(points={self.points}, bar_w={self.bar_white}, bar_b={self.bar_black}, "
            f"home_w={self.home_white}, home_b={self.home_black}, turn={self.turn})"
        )