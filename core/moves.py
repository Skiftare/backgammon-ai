"""Детерминированный валидатор и генератор легальных ходов назад.

Правила модели (важно для инж.):

- Среда **всегда симметризует** доску под текущего игрока перед генерацией:
  позиция поворачивается так, чтобы фишки ходящего стали «своими»
  (двигаются к индексу 0), независимо от цвета. Это упрощает правила и
  позволяет модели всегда видеть позицию со своей стороны.
- Легальный ход — это полная последовательность применения всех значений
  кубиков (`roll`), в произвольном порядке, если ход существует; если же
  есть только частично возможный ход — выбирается частичный (максимальный
  исчерпание кубиков).
- Генератор строит **все** легальные комбинации (с устранением дублей если
  одинаковый порядок применения), чтобы модель выбирала из них, а не
  генерировала невозможное.

Нотация хода:
`Move(steps=((from_point, die_value), ...))` — список применённых шагов.
`from_point` — индекс пункта (в симметризованных координатах 0..23);
`die_value` — какое число задействован (1..6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from .board import Position, N_POINTS

# Одна шашка при ходе (во flip-координатах): (from, die).
Step = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Move:
    steps: tuple[Step, ...]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"Move({self.steps})"


def _mirrored(pos: Position) -> Position:
    """Повернуть доску так, чтобы ходящий игрок был «белым» (движение к 0)."""
    # позиция под ходящим: flip её через current->future
    return pos.flipped()


def _block_target(value: int) -> bool:
    """Можно ли сделать ход в точку с этим содержимым (уже зеркальная сторона)."""
    # Здесь мы в нотации «ходящий игрок = +», значит точка с фишками соперника,
    # т.е. отрицательное - единственный блокер (>=1 соб ships занимают).
    if value == 0:
        return True
    # свой (со знаком ходящего) или одноий противника — можно бить, если один.
    # поскольку ходящий это "+" после зеркалирования, "противник" == отрицательное
    if value > 0:  # свои 1..5 — занято
        return True   # можно с толку: но только если есть место? занимается
    # value < 0: фишки соперника — пригоден если это БАР одна (бить), иначе блок
    return value == -1  # одну противника можно побить; >=2 нельзя


def _apply_step(pos: Position, fr: int, die: int) -> Position:
    """Применить один шаг (from, die) к симметрированной позе (with current player as white)."""
    pts = list(pos.points)
    v = pts[fr]
    if v <= 0:
        raise ValueError("нет своей фишки на пункте")
    # движение на die вниз (к индексу 0); из бара — fr == -1 спец.
    to = fr - die
    if to < 0:
        # выброс домой
        if not _can_bear_off(pos, fr):
            raise ValueError("нельзя выносить, фишки не в доме")
        pts[fr] = v - 1
        # home для игрока (белый — первый позиции): увеличить home_white
        pos = Position(points=tuple(pts), bar_white=pos.bar_white,
                       bar_black=pos.bar_black, home_white=pos.home_white + 1,
                       home_black=pos.home_black, turn=pos.turn)
        return pos
    if pos.points[to] < 0:
        # бить фишку противника: она на бар этого «игрока» (за доску) — в нашей
        # симметризации это бар соперника, но т.к. мы смотрим со стороны
        # ходящего, она отправляется на bar_opp... 
        pts[fr] = v - 1
        pts[to] = 1
        # enemy фишка на бар соперника (в Pos, соперник = чёрный if white):
        # у нас "белый-он-ход" держим br_white, но соперник чёрный: бар black
        pos2 = Position(points=tuple(pts), bar_white=pos.bar_white,
                        bar_black=pos.bar_black + 1, home_white=pos.home_white,
                        home_black=pos.home_black, turn=pos.turn)
        return pos2
    # обычное перемещ
    pts[to] = v
    pts[fr] = 0
    return Position(points=tuple(pts), bar_white=pos.bar_white,
                    bar_black=pos.bar_black, home_white=pos.home_white,
                    home_black=pos.home_black, turn=pos.turn)


def _can_bear(v: int) -> bool:
    """В симметрированный позе, можно вынести если пункт в доме (fr<6)."""
    return True  # решение в _apply_step: проверяем if to<0 and fr<6


def legal_moves(pos: Position, roll: Sequence[int]) -> list[Move]:
    """Все легальные полные ходы в симметрических координатах текущего игрока.

    Возвращает (уже симметризованные) Move. Применение к настоящей позиции —
    задаёт через wrap `apply_move_symmetric`.
    """
    mirrored = _mirrored(pos)   # где ходящий игрок белым, дом слева (0)
    all_moves = set()
    # рекурсивная генерация по исчерпанию оставшихся костей
    def dfs(cur: Position, remaining: Sequence[int], so_far: tuple[Step, ...]) -> None:
        # попробовать каждый кубик из remaining на каждой своей фишке
        steps_done = list(so_far)
        any_legal = False
        # перебор доступных пунктов (свои фишки: positive)
        for from_pt, val in cur.points:
            if val <= 0:
                continue
            for k, die in enumerate(remaining):
                to = from_pt - die
                if to < 0:
                    # если в доме (fr<6) - вынос
                    if from_pt < 6 and _can_bear(val):
                        nxt = _apply_step(cur, from_pt, die, exit=True)
                        any_legal = True
                        dfs(nxt, remaining[:k] + remaining[k+1:], so_far + ((from_pt, die),))
                else:
                    if not _block_target(cur.points[to]):
                        continue
                    nxt = _apply_step(cur, from_pt, die)
                    any_legal = True
                    dfs(nxt, remaining[:k] + remaining[k+1:], so_far + ((from_pt, die),))
        if not remaining or not any_legal:
            # сохранить для генерации всех полных (или partial-max)
            if not remaining or len(so_far) >= 1:  # ищет максимум ход с исчерпанием
                m = Move(so_far)
                if m.steps:
                    all_moves.add(m)

    dfs(mirrored, tuple(sorted(roll)), ())
    return sorted(all_moves, key=lambda m: (-len(m), m.steps))


def apply_sorted(pos: Position, move: Move) -> Position:
    """Применить симметризованный ход к оригинальной позиции.

    1) Используем генератор в нотации ходящего; результирующая позиция та же
       (флипы сочетаются), поэтому оставим детали в следующей функции.
    """
    raise NotImplementedError("применение хода к реальной позиции реализуется next")