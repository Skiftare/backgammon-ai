"""Среда коротких нард поверх MIT-движка (gym-backgammon).

Обёртка даёт наш API:
- `legal_moves(pos, roll) -> list[Move]` — все легальные полные ходы;
- `apply_move(pos, move) -> Position` — детерминированное применение;
- `new_game() -> Position`.

Движок: `core/_engine/backgammon.py` (MIT). Формат его состояния:
- board: list[24] of (count, color|None); WHITE=0 движется вверх (дом 18..23),
  BLACK=1 вниз (дом 0..5); bar=[bw,bb]; off=[ow,ob].
- action: кортеж (src, target), src может быть BAR, target может быть OFF.

Наша нотация (core/board.py) — ровно та же: белые=+, чёрные=-, дом 18..23 / 0..5.
Поэтому конвертация почти прямо: (count,color) -> +/- count, и обратно.
"""

from __future__ import annotations

from typing import Sequence

from .board import Position
from .dice import DiceRoll
from .moves import Move
from ._engine.backgammon import Backgammon, WHITE, BLACK, BAR, OFF, NUM_POINTS


def _pos_to_gym(pos: Position) -> Backgammon:
    """Собрать движок из нашего Position (fresh instance)."""
    g = Backgammon()
    # грязно: движок инициализируется стартовой доской по __init__; перезаписываем
    board = []
    for v in pos.points:
        if v > 0:
            board.append((v, WHITE))
        elif v < 0:
            board.append((-v, BLACK))
        else:
            board.append((0, None))
    g.board = board
    g.bar = [pos.bar_white, pos.bar_black]
    g.off = [pos.home_white, pos.home_black]
    # движок сам считает players_positions лениво; пересчитаем
    g.players_positions = g.get_players_positions()
    return g


def _gym_to_pos(g: Backgammon, turn: str = "white") -> Position:
    """Собрать наш Position из состояния движка."""
    pts = []
    for (cnt, color) in g.board:
        if color == WHITE:
            pts.append(cnt)
        elif color == BLACK:
            pts.append(-cnt)
        else:
            pts.append(0)
    return Position(
        points=tuple(pts),
        bar_white=g.bar[WHITE],
        bar_black=g.bar[BLACK],
        home_white=g.off[WHITE],
        home_black=g.off[BLACK],
        turn=turn,
    )


def new_game(turn: str = "white") -> Position:
    return Position.initial() if turn == "white" else Position.initial()


def legal_moves(pos: Position, roll: DiceRoll) -> list[Move]:
    """Все легальные полные ходы для текущего игрока при броске `roll`.

    Контракт: модель выбирает из этого списка; среда ничего невозможного не даст.
    """
    g = _pos_to_gym(pos)
    player = WHITE if pos.turn == "white" else BLACK
    actions = g.get_valid_plays(player, tuple(roll))
    moves: list[Move] = []
    for act in actions:
        # act — кортеж (src, target) для каждого кубика
        steps = tuple(_norm_step(s, t) for (s, t) in act if s is not None and t is not None)
        if steps:
            moves.append(Move(steps=steps))
    return moves


def _norm_step(src, target) -> tuple[int, int]:
    """Нормализованный шаг для нашего Move: (from_index|bar, to_index|off).

    Внутри движка src/target — индексы 0..23 или константы BAR/OFF.
    Мы храним метку как int: -1=BAR (приход с бара), 99=OFF (вынос), иначе индекс.
    """
    from_label = -1 if src == BAR else int(src)
    to_label = 99 if target == OFF else int(target)
    return (from_label, to_label)


def apply_move(pos: Position, move: Move) -> Position:
    """Применить ход и вернуть новую позицию (детерминированно)."""
    g = _pos_to_gym(pos)
    player = WHITE if pos.turn == "white" else BLACK
    # собрать action в формате движка
    action = []
    for (fr, to) in move.steps:
        src = BAR if fr == -1 else int(fr)
        target = OFF if to == 99 else int(to)
        action.append((src, target))
    g.execute_play(player, tuple(action))
    nxt = _gym_to_pos(g, turn="black" if pos.turn == "white" else "white")
    return nxt