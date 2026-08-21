#!/usr/bin/env python3
"""Сыграть с натренированным агентом (CLI).

Агент ходит greedy по value-сети; человек вводит ходы текстом.
Управление координатами — индексы 0..23 (как в нашей нотации) + b(бар)/o(вынос).

Использование:
    uv run python play.py --ckpt checkpoints/net_resnet.pt [--side black]

Человеческий ход (когда у вас бросок, напр. 3 и 5): введите разделённые
пробелом `from-to` шаги, например:
    12-15 16-19      (два шага)
    20-24 o          (один шаг + вынос с 20)
    b-1 16-19        (вход с бара на точку 1)

Правила: если агент сгенерил ход первым и ввёл ваш ход — надо вписать также
бросок цифрами (например `3 5`), затем ваш ход.
"""

from __future__ import annotations

import argparse
import random
import sys

import torch

from core.board import Position, TOTAL_CHECKERS
from core.features import Encoder
from core.game import legal_moves, apply_move, is_terminal
from model.net import make_value_net


def _show_board(pos: Position) -> None:
    def cell(p: int) -> str:
        v = pos.points[p]
        if v > 0:
            return f"+{v}"
        if v < 0:
            return f"-{abs(v)}"
        return " ."
    row = "  ".join(f"{i:>3}" for i in range(12, 24))
    print(f"\n      {row}")
    row2 = "  ".join(cell(i) for i in range(12, 24))
    print(f"  W→  {row2}   barW={pos.bar_white}")
    row3 = "  ".join(cell(i) for i in range(0, 12))
    print(f"  B→  {row3}   barB={pos.bar_black}   homeW={pos.home_white} homeB={pos.home_black}")
    print(f"  ход: {'БЕЛЫЕ' if pos.turn == 'white' else 'ЧЁРНЫЕ'}\n")


def _parse_move(pos: Position, text: str):
    # '12-15 16-19' / 'b-1' / 'o'
    steps = []
    for tok in text.split():
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "o":
            steps.append((-2, 99))  # special: вынос последней-точки сам решит
            continue
        if "-" not in tok:
            raise ValueError(f"ожидал from-to, получил {tok!r}")
        a, b = tok.split("-", 1)
        fr = 99 if a.lower() == "b" else int(a)
        to = 99 if b.lower() == "o" else int(b)
        steps.append((fr, to))
    return steps


import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/net_final.pt")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--side", default="black", choices=["white", "black"], help="за кого ходит человек")
    args = ap.parse_args()

    enc = Encoder()
    net = make_value_net(enc.dim(), hidden=args.hidden)
    net.eval()
    state = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(state)
    print(f"Загрузил {args.ckpt}")

    pos = Position.initial()
    human = args.side
    rng = random.Random()

    def agent_move(p: Position):
        a, b = rng.randint(1, 6), rng.randint(1, 6)
        roll = (a, a, a, a) if a == b else (a, b)
        print(f"[агент] бросок {roll[0] if len(roll) < 4 else roll[0]}+{'x4' if len(roll)==4 else ''} ({a},{b})")
        ms = legal_moves(p, roll)
        if not ms:
            print("[агент] нет хода — пропуск")
            return p
        # greedy по value
        best, bv = None, None
        for m in ms:
            nxt = apply_move(p, m)
            x = torch.tensor(enc.encode(nxt), dtype=torch.float32).unsqueeze(0)
            v = float(net.value(x).detach())
            if p.turn == "white":
                if best is None or v > bv:
                    best, bv = nxt, v
            else:
                if best is None or v < bv:
                    best, bv = nxt, v
        print("[агент] ходит:", list(best) if hasattr(best, 'steps') else best)
        return best

    while True:
        _show_board(pos)
        if is_terminal(pos):
            print("Игра окончена! Победил:", "white" if pos.home_white == 15 else "black")
            return

        if pos.turn == human:
            # человек вводит бросок и ход
            txt = input("Введи бросок (напр. '3 5' или дубль '4 4'): ").strip()
            if not txt:
                continue
            ds = [int(x) for x in txt.split()][:4]
            if len(ds) == 1:
                ds = [ds[0], ds[0], ds[0], ds[0]] if False else ds
            roll = tuple(ds) if len(ds) == 2 else (ds[0], ds[0], ds[0], ds[0])
            ms = legal_moves(pos, roll)
            if not ms:
                print("Нет легальных ходов при этом броске.")
                continue
            print("Доступные ходы (первые 5): ", [list(m.steps) for m in ms[:5]])
            htxt = input("Твой ход (from-to, пробел): ").strip()
            try:
                steps = _parse_move(pos, htxt)
            except ValueError as e:
                print("Некорректный формат:", e)
                continue
            # найти подходящий Move среди легальных (по шагам)
            match = None
            def keyeq(a, b):
                return sorted(list(a)) == sorted(list(b))
            for m in ms:
                if keyeq(m.steps, tuple(steps)):
                    match = m
                    break
            if match is None:
                print("Ход не из списка легальных.")
                continue
            pos = apply_move(pos, match)
        else:
            pos = agent_move(pos)


if __name__ == "__main__":
    main()