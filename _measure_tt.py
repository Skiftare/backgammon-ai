"""Оценка hit-rate транспозиционной таблицы при лукахеде (без взрыва памяти).

Прошлая версия разворачивала ПОЛНОЕ дерево depth=3 и улетела в OOM (SIGKILL) —
здесь всё забинджено: на каждом пласте родители субсемплятся до MAX_PARENTS,
узлов на пласт <= MAX_PARENTS*21*~10 ~ 300k. Плюс запускать через
~/bin/memrun 1536, чтобы RLIMIT_AS добил процесс СВОИМ лимитом, а не системой.

Отвечает на вопрос: в 1-пльном greedy value-кеш дал 0.01% хитов (бесполезен).
При лукахеде 2-3 плi дерево огромно, и если ТАМ повторы часты — транспозиционная
таблица (точный pack_exact-ключ) окупится. Если редки — кеш не нужен и на лукахеде.
"""
import random

from core.board import Position
from core.game import apply_move, legal_moves

ROLLS = [(a, b) for a in range(1, 7) for b in range(a, 7)]
MAX_PARENTS = 1500          # субсемпл уникальных родителей на пласт
MAX_CHILDREN_PLY = 400_000  # жёсткий потолок узлов на пласт (страховка)


def pack_exact(pos: Position) -> int:
    """Биективный ключ позиции (141 бит) — локальная копия из selfplay_gpu,
    чтобы НЕ импортировать torch (он тянет гигабайты VA при импорте)."""
    k = 0
    for v in pos.points:
        k = (k << 5) | (v + 15)
    k = (k << 5) | pos.bar_white
    k = (k << 5) | pos.bar_black
    k = (k << 5) | pos.home_white
    k = (k << 5) | pos.home_black
    return (k << 1) | (0 if pos.turn == "white" else 1)


def expand(parents, rng):
    """Родители -> (дети, всего_узлов, уникальных). Дети потом субсемплятся."""
    children = []
    for p in parents:
        for roll in ROLLS:
            for m in legal_moves(p, roll):
                children.append(apply_move(p, m))
                if len(children) >= MAX_CHILDREN_PLY:
                    break
            if len(children) >= MAX_CHILDREN_PLY:
                break
        if len(children) >= MAX_CHILDREN_PLY:
            break
    keys = [pack_exact(c) for c in children]
    uniq = set(keys)
    return children, len(children), len(uniq)


rng = random.Random(1)
seeds = [Position.initial()]
for _ in range(8):
    p = Position.initial()
    for _ in range(rng.randint(5, 25)):
        roll = rng.choice(ROLLS)
        ms = legal_moves(p, roll)
        if ms:
            p = apply_move(p, rng.choice(ms))
    seeds.append(p)

frontier = seeds
print("ply | узлов | уникальных | dup-rate | (родителей после субсемпла)")
for ply in range(1, 5):
    children, n_nodes, n_uniq = expand(frontier, rng)
    dup = n_nodes - n_uniq
    print(f"  {ply} | {n_nodes:7d} | {n_uniq:7d} | {100 * dup / max(1, n_nodes):5.1f}% "
          f"| {len(frontier)}")
    # субсемпл уникальных до следующего пласта
    key_to_pos = {}
    for c in children:
        key_to_pos.setdefault(pack_exact(c), c)
    uniq_children = list(key_to_pos.values())
    if len(uniq_children) > MAX_PARENTS:
        rng.shuffle(uniq_children)
        uniq_children = uniq_children[:MAX_PARENTS]
    frontier = uniq_children

print("\ndup-rate пласта = доля узлов, которые уже встречались в этом пласте")
print("=> если dup-rate падает к 0, TT на лукахеде почти не поможет;")
print("=> если растёт — позиции схлопываются, таблица окупится.")