"""Тесты среды поверх MIT-движка правил."""

from core.board import Position
from core.engine import new_game, legal_moves, apply_move
from core.moves import Move
from core._engine.backgammon import Backgammon, init_board


def test_initial_matches_engine_init():
    """Наш старт должен совпадать с init_board() движка."""
    g = Backgammon()
    ref = g.board  # list[(cnt,color)]
    pos = Position.initial()
    assert pos.at(5) == 5 and pos.at(7) == 3 and pos.at(12) == 5 and pos.at(23) == 2
    assert pos.at(0) == -2 and pos.at(11) == -5 and pos.at(16) == -3 and pos.at(18) == -5
    # сравним полностью с доски движка
    for i, (cnt, color) in enumerate(ref):
        if color == 0:  # WHITE
            assert pos.at(i) == cnt
        elif color == 1:
            assert pos.at(i) == -cnt
        else:
            assert pos.at(i) == 0


def test_legal_moves_nonempty_on_start():
    """На старте с любым броском у белых есть легальный ход."""
    moves = legal_moves(Position.initial(), (1, 1))
    assert isinstance(moves, list) and len(moves) >= 1


def test_apply_move_changes_turn_and_keeps_invariants():
    pos = Position.initial()
    moves = legal_moves(pos, (3, 1))
    assert moves, "есть ходы на 3-1"
    nxt = apply_move(pos, moves[0])
    assert nxt.turn == "black"
    assert nxt.check_invariants()


def test_deterministic_apply():
    pos = Position.initial()
    moves = legal_moves(pos, (4, 2))
    for m in moves:
        a = apply_move(pos, m)
        b = apply_move(pos, m)
        assert a == b  # детерминизм