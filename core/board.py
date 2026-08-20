"""Позиция нардов: 24 пункта с фишками, бар (сбитые), дом (home), очередь.

Представление:
- `points`: кортеж из 24 элементов (индекс 0..23). Точка представлена
  количеством фишек со знаком: `+n` — n белых, `-n` — n чёрных, 0 — пусто.
- Белый движется к своему дому на индексе 0 (выход за левую грань), чёрный —
  к индексу 23 (правая грань).
- `bar` — сбитые фишки (за доской), отдельно белым/чёрным.
- `home` — фишки, уже выведенные (победа при всех home = total 15).

Позиция неизменяема (frozen dataclass) — каждый ход создаёт новое состояние.
Это важно для валидатора и для ветвления в поиске (MCTS и т.п.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

N_POINTS = 24
TOTAL_CHECKERS = 15


@dataclass(frozen=True, slots=True)
class Position:
    points: tuple[int, ...] = field(default_factory=lambda: (0,) * N_POINTS)
    bar_white: int = 0
    bar_black: int = 0
    home_white: int = 0
    home_black: int = 0
    turn: str = "white"  # 'white' или 'black'

    def __post_init__(self) -> None:
        if len(self.points) != N_POINTS:
            raise ValueError(f"ожидалось {N_POINTS} пунктов, получено {len(self.points)}")
        if self.turn not in ("white", "black"):
            raise ValueError(f"неизвестный игрок: {self.turn!r}")

    @classmethod
    def initial(cls) -> "Position":
        """Классическая стартовая расстановка.

        С белой стороны (положительные): 2 на 24, 5 на 13, 3 на 8, 5 на 6.
        С чёрной (отрицательные, зеркально): 2 на 1, 5 на 12, 3 на 17, 5 на 19.
        Индексы (0-based): 24→23, 13→12, 8→7, 6→5; 1→0, 12→11, 17→16, 19→18.
        """
        pts = [0] * N_POINTS
        for idx, cnt in ((23, 2), (12, 5), (7, 3), (5, 5)):
            pts[idx] = cnt          # белые
        for idx, cnt in ((0, -2), (11, -5), (16, -3), (18, -5)):
            pts[idx] = cnt          # чёрные
        return cls(points=tuple(pts), turn="white")

    # ---------- удобство ----------

    def at(self, point: int) -> int:
        """Значение на пункте (0..23)."""
        return self.points[point]

    def count_white(self) -> int:
        return sum(max(0, p) for p in self.points) + self.bar_white + self.home_white

    def count_black(self) -> int:
        return sum(max(0, -p) for p in self.points) + self.bar_black + self.home_black

    def check_invariants(self) -> bool:
        """Проверка консервации фишек (15 на игрока суммарно)."""
        return self.count_white() == TOTAL_CHECKERS and self.count_black() == TOTAL_CHECKERS

    def iter_points(self) -> Iterator[tuple[int, int]]:
        """(point_index, value) для ненулевых пунктов."""
        for i, v in enumerate(self.points):
            if v:
                yield i, v

    def flipped(self) -> "Position":
        """Повёрнутая позиция для игры с противоположного ракурса.

        Используется, чтобы модель видела позицию всегда «со своей стороны»
        (белые фишки со знаком +, движение вниз), независимо от того, чей ход.
        """
        rev = [0] * N_POINTS
        for i in range(N_POINTS):
            rev[N_POINTS - 1 - i] = -self.points[i]
        new_turn = "white" if self.turn == "white" else "white"
        return Position(
            points=tuple(rev),
            bar_white=self.bar_black,
            bar_black=self.bar_white,
            home_white=self.home_black,
            home_black=self.home_white,
            turn="white",
        )

    def __repr__(self) -> str:
        return (
            f"Position(points={self.points}, bar_w={self.bar_white}, bar_b={self.bar_black}, "
            f"home_w={self.home_white}, home_b={self.home_black}, turn={self.turn})"
        )