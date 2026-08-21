#!/usr/bin/env python3
"""Веб-интерфейс: игра в нарды против агента через canvas-доску.

Endpoints (JSON):
  GET  /             — страница (static/index.html)
  POST /api/new       — новая игра {side: 'white'|'black'} (человек за side)
  POST /api/roll      — бросить кости (для человеческого хода) -> {roll, legal_moves}
  POST /api/legals    — {roll} -> легальные ходы (для подсветки)
  POST /api/move      — {from,to} применить шаг человека (validator движком)
  POST /api/agent     — агент ходит greedy (value-net) или random, если torch нет

Состояние игры хранится на клиенте (frontend держит позицию и шлёт `pos` в каждом
запросе — сервер stateless). Агент: если torch установлен и есть чекпоинт — greedy
по value-сети, иначе случайный (чтобы доска/клики работали даже без ML).

Запуск: uv run flask --app web.app run --port 8000
"""

from __future__ import annotations

import json
import random

from flask import Flask, jsonify, request, send_from_directory

from core.board import Position, TOTAL_CHECKERS
from core.game import legal_moves, apply_move, is_terminal
from core.rolls import roll  # честный бросок

app = Flask(__name__, static_folder="static")

# ── агент: value-сеть, если torch есть ──
NET = None
ENC = None
CKPT = "checkpoints/net_final.pt"


def _load_net():
    global NET, ENC
    try:
        import torch
        from core.features import Encoder
        from model.net import make_value_net
        ENC = Encoder()
        NET = make_value_net(ENC.dim())
        NET.eval()
        state = torch.load(CKPT, map_location="cpu")
        NET.load_state_dict(state)
        return True
    except Exception as e:
        print(f"[agent] value-net недоступна ({e}); использую random", flush=True)
        return False


_loaded = False


def pos_to_json(pos: Position) -> dict:
    return {
        "points": list(pos.points),
        "bar_white": pos.bar_white,
        "bar_black": pos.bar_black,
        "home_white": pos.home_white,
        "home_black": pos.home_black,
        "turn": pos.turn,
        "terminal": is_terminal(pos),
    }


def pos_from_json(d: dict) -> Position:
    return Position(points=tuple(d["points"]), bar_white=d["bar_white"], bar_black=d["bar_black"],
                    home_white=d["home_white"], home_black=d["home_black"], turn=d["turn"])


def _roll_json():
    a, b = roll(), roll()
    return (a, a, a, a) if a == b else (a, b)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/new")
def new_game():
    side = request.json.get("side", "black")
    return jsonify({"pos": pos_to_json(Position.initial()), "side": side})


@app.post("/api/roll")
def do_roll():
    d = request.json
    pos = pos_from_json(d["pos"])
    r = tuple(d.get("roll")) if d.get("roll") else _roll_json()
    moves = legal_moves(pos, r)
    return jsonify({
        "pos": pos_to_json(pos),
        "roll": list(r),
        "legal_moves": [list(m.steps) for m in moves],
    })


@app.post("/api/legals")
def legals():
    d = request.json
    pos = pos_from_json(d["pos"])
    r = tuple(d["roll"])
    moves = legal_moves(pos, r)
    return jsonify({"legal_moves": [list(m.steps) for m in moves]})


@app.post("/api/move")
def do_move():
    d = request.json
    pos = pos_from_json(d["pos"])
    rollv = tuple(d["roll"])
    steps = [tuple(s) for s in d["steps"]]
    # найти легальный Move с этими шагами
    target = None
    for m in legal_moves(pos, rollv):
        if sorted(list(m.steps)) == sorted(steps):
            target = m
            break
    if target is None:
        return jsonify({"error": "нелегальный ход"}), 400
    nxt = apply_move(pos, target)
    return jsonify({"pos": pos_to_json(nxt)})


@app.post("/api/agent")
def agent_step():
    global _loaded, NET, ENC
    d = request.json
    pos = pos_from_json(d["pos"])
    r = tuple(d.get("roll")) or _roll_json()
    moves = legal_moves(pos, r)
    if not moves:
        # пропуск
        nxt = Position(points=pos.points, bar_white=pos.bar_white, bar_black=pos.bar_black,
                       home_white=pos.home_white, home_black=pos.home_black,
                       turn="black" if pos.turn == "white" else "white")
        return jsonify({"pos": pos_to_json(nxt), "roll": list(r), "moved": False})

    if NET is None:
        if not _loaded:
            _loaded = _load_net()
            if NET is None:
                pass
        if NET is None:
            # random агент
            m = random.choice(moves)
            nxt = apply_move(pos, m)
            return jsonify({"pos": pos_to_json(nxt), "roll": list(r), "moved": True, "agent": "random"})

    # greedy по value
    import torch
    best, bv = None, None
    for m in moves:
        cand = apply_move(pos, m)
        x = torch.tensor(ENC.encode(cand), dtype=torch.float32).unsqueeze(0)
        v = float(NET.value(x).detach())
        if pos.turn == "white":
            if best is None or v > bv:
                best, bv = cand, v
        else:
            if best is None or v < bv:
                best, bv = cand, v
    return jsonify({"pos": pos_to_json(best), "roll": list(r), "moved": True, "agent": "value"})