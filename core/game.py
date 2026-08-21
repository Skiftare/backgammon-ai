"""Собственный движок коротких нард (детерминированный, читаемый).

Нотация — из `core/board.py`: белые = +N, двигаются к своим 18..23 (дом),
чёрные = -N, двигаются к своим 0..5 (дом). bar_white / bar_black — сбитые;
home_white / home_black — вынесенные.

Основные правила, реализованные здесь:
- Направление: белые p -> p + step; чёрные p -> p - step.
- Вход с бара: белый входит в 24-строчную точку ⠀(индекс 23-(step-1)), чёрный —
  в индекс (step-1).
- Приземление запрещено на точку с >=2 фишками соперника (закрытая).
- Битьё: приземление на блот соперника (ровно 1) — его фишка на бар.
- Вынос: разрешён только когда все фишки игрока в его доме; кость n выносит
  фишку с точки n (или по правилу «все ниже» — с самой высокой).
- Дубль даёт 4 одинаковые кости.
- Обязующее правило: если фишки на баре — первым делом вход (максимум костей
  на вход, остаток — обычные ходы).
"""

from __future__ import annotations

from .board import N_POINTS, TOTAL_CHECKERS, Position
from .dice import DiceRoll
from .moves import BAR, OFF, Move

STEP = tuple[int, int]  # (from_label, to_label): -1 = bar, 99 = off, иначе index


def _dir(turn: str) -> int:
    return 1 if turn == "white" else -1


def _own(pos: Position, p: int) -> int:
    """Число фишек текущего игрока на точке p (положительное)."""
    v = pos.points[p]
    if _dir(pos.turn) == 1:
        return max(0, v)
    return -v if v < 0 else 0


def _bar(pos: Position) -> int:
    return pos.bar_white if pos.turn == "white" else pos.bar_black


def _landable(pos: Position, p: int) -> bool:
    """Можно ли встать на точку p (не закрыта соперником)."""
    v = pos.points[p]
    d = _dir(pos.turn)
    return (v >= -1) if d == 1 else (v <= 1)


def _entry_point(turn: str, step: int) -> int:
    return 23 - (step - 1) if turn == "white" else step - 1


def _home(turn: str, p: int) -> bool:
    return (18 <= p <= 23) if turn == "white" else (0 <= p <= 5)


def _all_in_home(pos: Position) -> bool:
    if _bar(pos) > 0:
        return False
    d = _dir(pos.turn)
    for p, v in enumerate(pos.points):
        if d == 1 and v > 0 and not _home(pos.turn, p):
            return False
        if d == -1 and v < 0 and not _home(pos.turn, p):
            return False
    return True


def _apply_step(pos: Position, step: STEP) -> Position:
    """Применить один атомарный шаг; возвращает состояние с hit/off (без смены хода)."""
    pts = list(pos.points)
    fr, to = step
    d = _dir(pos.turn)

    # пересчитаем бары/дома для текущего и соперника
    bar_me = pos.bar_white if pos.turn == "white" else pos.bar_black
    bar_opp = pos.bar_black if pos.turn == "white" else pos.bar_white
    home_me = pos.home_white if pos.turn == "white" else pos.home_black
    home_opp = pos.home_black if pos.turn == "white" else pos.home_white

    def set_own(p: int, delta: int) -> None:
        if d == 1:
            pts[p] += delta
        else:
            pts[p] -= delta

    def own_at(p: int) -> int:
        return pts[p] if d == 1 else -pts[p]

    if fr == BAR:
        if bar_me <= 0:
            raise ValueError("нет фишки на баре")
        bar_me -= 1
        if to < 0 or to >= N_POINTS:
            raise ValueError("вход с бара вне доски")
        if d == 1 and pts[to] == -1 or d == -1 and pts[to] == 1:
            bar_opp += 1
            pts[to] = 0
        set_own(to, +1)
    elif to == OFF:
        if _all_in_home(pos) is False:
            raise ValueError("вынос запрещён: не все фишки в доме")
        if own_at(fr) <= 0:
            raise ValueError("нет фишки для выноса")
        set_own(fr, -1)
        home_me += 1
    else:
        if fr < 0 or fr >= N_POINTS or to < 0 or to >= N_POINTS:
            raise ValueError("шаг вне доски")
        if own_at(fr) <= 0:
            raise ValueError("нет фишки на source")
        # если на to блот соперника — бить
        if d == 1 and pts[to] == -1 or d == -1 and pts[to] == 1:
            bar_opp += 1
            pts[to] = 0
        set_own(fr, -1)
        set_own(to, +1)

    return Position(
        points=tuple(pts),
        bar_white=(bar_me if pos.turn == "white" else bar_opp),
        bar_black=(bar_opp if pos.turn == "white" else bar_me),
        home_white=(home_me if pos.turn == "white" else home_opp),
        home_black=(home_opp if pos.turn == "white" else home_me),
        turn=pos.turn,
    )


def _single_moves(pos: Position, step: int) -> list[STEP]:
    """Все атомарные шаги по одной кости (для текущего игрока)."""
    out: list[STEP] = []
    # вход с бара
    if _bar(pos) > 0:
        p = _entry_point(pos.turn, step)
        if _landable(pos, p):
            out.append((BAR, p))
        return out  # при фишках на баре — только вход
    d = _dir(pos.turn)
    for p in range(N_POINTS):
        if _own(pos, p) <= 0:
            continue
        tgt = p + d * step
        # вынос: только когда все фишки в доме и p в доме.
        # Правило: кость n выносит точку p, если её номинал <= n, И на всех точках
        # с номиналом > n нет своих фишек (иначе должен играть старшую).
        if (tgt >= N_POINTS) if d == 1 else (tgt < 0):
            if _all_in_home(pos) and _home(pos.turn, p):
                nominal = (24 - p) if d == 1 else (p + 1)
                if step >= nominal:
                    blocked = False
                    for q in range(N_POINTS):
                        if q == p:
                            continue
                        if _own(pos, q) <= 0:
                            continue
                        q_nom = (24 - q) if d == 1 else (q + 1)
                        if _home(pos.turn, q) and q_nom > nominal:
                            blocked = True
                            break
                    if not blocked:
                        out.append((p, OFF))
        elif 0 <= tgt < N_POINTS and _landable(pos, tgt):
            out.append((p, tgt))
    return out


def legal_moves(pos: Position, roll: DiceRoll) -> list[Move]:
    """Все полные легальные ходы (максимального числа костей) для текущего игрока.

    DFS: применяем кости в любом порядке; на каждом узле перебираем все атомарные
    шаги для этой кости; собираем пути. В конце выбираем только максимальные по
    числу применённых костей (правило обязательности/максимизации).
    """
    steps_all = list(roll)  # уже развёрнут в кортеж (2 или 4)
    results: list[tuple[STEP, ...]] = []

    def dfs(pos_: Position, rem: list[int], path: tuple[STEP, ...]) -> None:
        # если остались кости, и никакой шаг невозможен — заканчиваем (частичный)
        if not rem:
            if path:
                results.append(path)
            return
        any_ok = False
        for i, st in enumerate(rem):
            for mv in _single_moves(pos_, st):
                any_ok = True
                nxt = _apply_step(pos_, mv)
                new_rem = list(rem)
                new_rem.pop(i)
                dfs(nxt, new_rem, path + (mv,))
        if not any_ok:
            if path:
                results.append(path)

    dfs(pos, steps_all, ())
    # дедуп: порядок шагов ВАЖЕН (это последовательность применения: следующий шаг
    # строится на результате предыдущего), поэтому ключ — сам кортеж, не sorted.
    uniq: dict[tuple[STEP, ...], None] = {}
    for r in results:
        uniq.setdefault(r, None)
    best = sorted(uniq.keys(), key=len, reverse=True)
    maxlen = len(best[0]) if best else 0
    best = [r for r in best if len(r) == maxlen]
    return [Move(steps=tuple(r)) for r in best]


def apply_move(pos: Position, move: Move) -> Position:
    """Применить легальный ход, вернуть новую позицию со сменой хода."""
    p = pos
    for st in move.steps:
        p = _apply_step(p, st)
    return Position(
        points=p.points,
        bar_white=p.bar_white,
        bar_black=p.bar_black,
        home_white=p.home_white,
        home_black=p.home_black,
        turn="black" if pos.turn == "white" else "white",
    )


def is_terminal(pos: Position) -> bool:
    return pos.home_white == TOTAL_CHECKERS or pos.home_black == TOTAL_CHECKERS