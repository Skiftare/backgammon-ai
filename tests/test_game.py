"""Базовые тесты собственного движка правил (core/game.py).

Статус движка: стабильное ядро (генерация + применение, суммы фишек ок на
большинстве партий). Известный TODO: bearing-фаза (вынос с переполнением) —
3/20 партий теряют фишку/провисают; полнота правил в docs/backgammon_rules_spec.md.
"""

import random

from core.board import Position
from core.game import legal_moves, apply_move, is_terminal, _apply_step


def test_legal_moves_nonempty_on_start():
    pos = Position.initial()
    for roll in [(3, 1), (4, 2), (6, 5), (1, 1), (4, 4)]:
        assert legal_moves(pos, roll), f"на старте для {roll} есть ходы"


def test_apply_keeps_invariants_on_start_moves():
    pos = Position.initial()
    for roll in [(3, 1), (5, 4), (2, 1), (6, 5)]:
        for m in legal_moves(pos, roll):
            nxt = apply_move(pos, m)
            assert nxt.check_invariants()
            assert nxt.turn != pos.turn


def test_apply_step_preserves_total():
    """Суммы фишек обоих игроков сохраняются на всех применяемых шагах (jаkr - no loss)."""
    import random as _r
    rng = _r.Random(9)
    pos = Position.initial()
    for _ in range(60):
        a, b = rng.randint(1, 6), rng.randint(1, 6)
        roll = (a, a, a, a) if a == b else (a, b)
        ms = legal_moves(pos, roll)
        if not ms:
            pos = Position(points=pos.points, bar_white=pos.bar_white, bar_black=pos.bar_black,
                           home_white=pos.home_white, home_black=pos.home_black,
                           turn="black" if pos.turn == "white" else "white")
            continue
        pos = apply_move(pos, rng.choice(ms))
        assert pos.check_invariants(), "суммы 15/15 должны сохраняться"


def test_terminal():
    from core.board import Position as P
    p = P(points=(0,) * 24, home_white=15, turn="white")
    assert is_terminal(p)