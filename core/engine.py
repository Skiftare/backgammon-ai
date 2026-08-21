"""Обёртка движка: НАША механика применения хода (правильная, с bar/hit/off).

Генерация легальных ходов — MIT-движок gym-backgammon (`get_valid_plays`),
применение — здесь, по шагам, с учётом:
- вход с бара (fr=-1), битьё (блот соперника → бар), вынос (to=99);
- **порядок применения шагов — по правилам, т.е. любые перестановки костей
  допустимы, пока промежуточные точки легальны**. Движок отдаёт chain-шаги
  (одна фишка на две кости) в произвольном порядке, поэтому мы пробуем все
  перестановки и применяем первую валидную. Это соответствует правилам:
  кости играются в любом разрешённом порядке.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

from .board import Position, TOTAL_CHECKERS
from .dice import DiceRoll
from .moves import Move, BAR, OFF
from ._engine.backgammon import Backgammon, WHITE, BLACK, NUM_POINTS


def legal_moves(pos: Position, roll: DiceRoll) -> list[Move]:
    """Все легальные полные ходы для текущего игрока (генерация движком)."""
    g = _pos_to_gym(pos)
    player = WHITE if pos.turn == "white" else BLACK
    actions = g.get_valid_plays(player, tuple(roll))
    moves: list[Move] = []
    for act in actions:
        steps = tuple(_norm_step(s, t) for (s, t) in act if s is not None and t is not None)
        if steps:
            moves.append(Move(steps=steps))
    return moves


def _norm_step(src, target) -> tuple[int, int]:
    from_l = -1 if _is_bar(src) else int(src)
    to_l = 99 if _is_off(target) else int(target)
    return (from_l, to_l)


def _is_bar(v) -> bool:
    return v == "bar" or v == BAR or v == -1


def _is_off(v) -> bool:
    return v == "off" or v == OFF or v == 99


def _pos_to_gym(pos: Position) -> Backgammon:
    g = Backgammon()
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
    g.players_positions = g.get_players_positions()
    return g


def _gym_to_pos(g: Backgammon, turn: str) -> Position:
    pts = []
    for (cnt, color) in g.board:
        if color == WHITE:
            pts.append(cnt)
        elif color == BLACK:
            pts.append(-cnt)
        else:
            pts.append(0)
    return Position(points=tuple(pts), bar_white=g.bar[WHITE], bar_black=g.bar[BLACK],
                    home_white=g.off[WHITE], home_black=g.off[BLACK], turn=turn)


def _try_apply_seq(g: Backgammon, player: int, steps: Sequence[tuple[int, int]]) -> bool:
    """Применить последовательность шагов к доске; True если все легальны."""
    opp = 1 - player
    for (fr, to) in steps:
        if fr == -1:  # вход с бара
            if g.bar[player] <= 0:
                return False
            g.bar[player] -= 1
            cnt, color = g.board[to]
            if color == opp and cnt == 1:
                g.bar[opp] += 1
                cnt = 0
            g.board[to] = (cnt + 1, player)
        elif to == 99:  # вынос
            cnt, color = g.board[fr]
            if color != player or cnt <= 0:
                return False
            cnt -= 1
            g.board[fr] = (cnt, player if cnt > 0 else None)
            g.off[player] += 1
        else:  # обычное перемещение
            if to < 0 or to >= NUM_POINTS:
                return False
            cnt, color = g.board[fr]
            if color != player or cnt <= 0:
                return False
            cnt -= 1
            g.board[fr] = (cnt, player if cnt > 0 else None)
            c2, c2c = g.board[to]
            if c2c == opp and c2 == 1:
                g.bar[opp] += 1
                c2 = 0
            g.board[to] = (c2 + 1, player)
    return True


def apply_move(pos: Position, move: Move) -> Position:
    """Применить легальный ход, вернуть новую позицию (детерминированно).

    Стратегия-гибрид:
    1) пробуем их `execute_play` (движок сам валиден для сгенерированных ходов);
    2) если он кидает AssertionError (известный баг на bar-входе в комбо) —
       применяем нашей перестановочной механикой (правила с bar/hit/off);
    3) финальная проверка инвариантов (15/15).
    """
    player = WHITE if pos.turn == "white" else BLACK
    steps = list(move.steps)
    action = [(BAR if fr == -1 else fr, OFF if to == 99 else to) for (fr, to) in steps]

    # 1) их execute_play
    g = _pos_to_gym(pos)
    ok = False
    try:
        g.execute_play(player, tuple(action))
        ok = True
    except AssertionError:
        ok = False
    if not ok:
        # 2) fallback: наши перестановки
        applied = False
        for perm in permutations(steps):
            gg = _pos_to_gym(pos)
            if _try_apply_seq(gg, player, perm):
                g = gg
                applied = True
                break
        if not applied:
            raise AssertionError(f"не удалось применить Move({steps}) из {pos.turn}")
    else:
        # пересчитать players_positions (execute_play уже сделал, но для чистоты)
        g.players_positions = g.get_players_positions()

    # 3) инварианты
    w = sum(c for (c, p) in g.board if p == WHITE)
    b = sum(c for (c, p) in g.board if p == BLACK)
    if not (w + g.bar[WHITE] + g.off[WHITE] == TOTAL_CHECKERS and
            b + g.bar[BLACK] + g.off[BLACK] == TOTAL_CHECKERS):
        raise AssertionError(
            f"инвариант: w={w}+{g.bar[WHITE]}+{g.off[WHITE]} b={b}+{g.bar[BLACK]}+{g.off[BLACK]}")
    return _gym_to_pos(g, turn="black" if pos.turn == "white" else "white")


def new_game(turn: str = "white") -> Position:
    return Position.initial()