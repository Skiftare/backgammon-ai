#!/usr/bin/env python3
"""Веб-интерфейс: игра в нарды против агента через canvas-доску.

Endpoints (JSON):
  GET  /                  — страница (static/index.html)
  GET  /vendor/<p>        — статика bgboard (css/js/img)
  POST /api/new           — {side} новая игра -> {game_id, pos, side}
  POST /api/sync          — {game_id} восстановить после перезагрузки
  POST /api/roll          — {game_id} бросок человека
  POST /api/steps         — {game_id} легальные атомарные ходы под тек. кость
  POST /api/step          — {game_id, from, to} шаг человека
  POST /api/agent_roll    — {game_id} бросок агента (возвращает его кость)
  POST /api/agent_move    — {game_id} агент применяет ход
  POST /api/meta          — сведения о «сопернике» (сеть, обучение, чекпоинт)

Состояние партии хранится НА СЕРВЕРЕ (в dict по game_id) — перезагрузка страницы
не может отменить/повторить ход или перебросить кости: бросок и ходы привязаны
к партии. Агент отдельно «выбрасывает» кости (agent_roll) и отдельно ходит
(agent_move), что позволяет фронту показывать кости соперника и паузу «думает».

Запуск: uv run flask --app web.app run --port 8000
"""

from __future__ import annotations

import os
import random

from flask import Flask, jsonify, request, send_from_directory

from core.board import Position
from core.game import _apply_step, _single_moves, apply_move, is_terminal, legal_moves
from core.rolls import roll  # честный бросок

app = Flask(__name__, static_folder="static")

# ── метаданные «соперника» ──
CKPT = "checkpoints/net_final.pt"
_META_BASE = {
    "model": "ValueNet (MLP, стиль TD-Gammon / Tesauro)",
    "architecture": "293 → 128 → 128 → 1 (Tanh; value-only, greedy-выбор)",  # ponytail: hardcode 128 = дефолт make_value_net
    "training": "self-play greedy + TD(lambda) (lambda=0.7, lr=1e-3, gamma=1.0)",
    "iterations": "1000 партий (net_1000.pt -> net_final.pt)",
    "checkpoint": os.path.abspath(CKPT),
    "net": "value-net ⟶ не загружено (random fallback)",
}


# ── агент: value-сеть, если torch есть ──
NET = None
ENC = None


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
        _META_BASE["net"] = "value-net (" + os.path.abspath(CKPT) + ")"
        return True
    except Exception as e:
        print(f"[agent] value-net недоступна ({e}); использую random", flush=True)
        _META_BASE["net"] = f"random fallback (value-net недоступна: {e})"
        return False


_loaded = False


def _agent_choice(pos: Position, dice: tuple[int, ...]):
    """Возвращает (новая_позиция_или_None, kind, list_of_steps)."""
    global _loaded, NET, ENC
    moves = legal_moves(pos, dice)
    if not moves:
        return None, "none", []
    if NET is None:
        if not _loaded:
            _loaded = _load_net()
        if NET is None:
            m = random.choice(moves)
            return apply_move(pos, m), "random", [list(s) for s in m.steps]
    import torch
    best, bv, best_m = None, None, None
    for m in moves:
        cand = apply_move(pos, m)
        x = torch.tensor(ENC.encode(cand), dtype=torch.float32).unsqueeze(0)
        v = float(NET.value(x).detach())
        if pos.turn == "white":
            if best is None or v > bv:
                best, bv, best_m = cand, v, m
        else:
            if best is None or v < bv:
                best, bv, best_m = cand, v, m
    return best, "value", [list(s) for s in best_m.steps]


# ── сервер stateless: состояние на клиенте, сервер валидирует ──


def pos_to_json(pos: Position) -> dict:
    winner = None
    if is_terminal(pos):
        winner = "white" if pos.home_white == 15 else ("black" if pos.home_black == 15 else None)
    return {
        "points": list(pos.points),
        "bar_white": pos.bar_white,
        "bar_black": pos.bar_black,
        "home_white": pos.home_white,
        "home_black": pos.home_black,
        "turn": pos.turn,
        "terminal": winner is not None,
        "winner": winner,
    }


def pos_from_json(d: dict) -> Position:
    return Position(points=tuple(d["points"]), bar_white=d["bar_white"], bar_black=d["bar_black"],
                    home_white=d["home_white"], home_black=d["home_black"], turn=d["turn"])


def _switch(pos: Position) -> Position:
    return Position(points=pos.points, bar_white=pos.bar_white, bar_black=pos.bar_black,
                    home_white=pos.home_white, home_black=pos.home_black,
                    turn="black" if pos.turn == "white" else "white")


def _roll_json():
    a, b = roll(), roll()
    return (a, a, a, a) if a == b else (a, b)


def _any_playable(pos: Position, rem: list[int]) -> bool:
    return any(_single_moves(pos, d) for d in rem)


@app.after_request
def _log(resp):
    body = request.get_data(cache=True, parse_form_data=False) or b""
    print(
        f"{request.method} {request.path} -> {resp.status_code} "
        f"{body[:200].decode('utf-8', 'replace')}",
        flush=True,
    )
    return resp


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/vendor/<path:p>")
def vendor(p):
    return send_from_directory("static/vendor", p)


@app.post("/api/new")
def new_game():
    return jsonify({"pos": pos_to_json(Position.initial())})


@app.post("/api/sync")
def sync():
    d = request.json
    pos = pos_from_json(d["pos"])
    return jsonify({"pos": pos_to_json(pos), "meta": _META_BASE})


@app.post("/api/roll")
def do_roll():
    d = request.json
    pos = pos_from_json(d["pos"])
    if d.get("roll"):
        return jsonify({"error": "кости уже брошены"}), 400
    dl = _roll_json()
    if not legal_moves(pos, dl):
        nxt = _switch(pos)
        return jsonify({"pos": pos_to_json(nxt), "roll": None, "turned": True})
    return jsonify({"pos": pos_to_json(pos), "roll": list(dl), "turned": False})


@app.post("/api/steps")
def steps():
    d = request.json
    pos = pos_from_json(d["pos"])
    rem = list(d.get("roll") or [])
    seen = set()
    out = []
    for die in rem:
        for (fr, to) in _single_moves(pos, die):
            key = (fr, to)
            if key not in seen:
                seen.add(key)
                out.append([fr, to])
    return jsonify({"steps": out})


@app.post("/api/step")
def do_step():
    d = request.json
    pos = pos_from_json(d["pos"])
    rem = list(d["roll"])
    fr, to = int(d["from"]), int(d["to"])
    used = -1
    for i, die in enumerate(rem):
        if (fr, to) in _single_moves(pos, die):
            used = i
            break
    if used == -1:
        return jsonify({"error": "нелегальный ход"}), 400
    nxt = _apply_step(pos, (fr, to))
    rem.pop(used)
    done = not _any_playable(nxt, rem)
    if done:
        nxt = _switch(nxt)
        return jsonify({"pos": pos_to_json(nxt), "roll": None, "done": True})
    return jsonify({"pos": pos_to_json(nxt), "roll": rem, "done": False})


@app.post("/api/agent_roll")
def agent_roll():
    d = request.json
    pos = pos_from_json(d["pos"])
    if d.get("roll"):
        return jsonify({"pos": pos_to_json(pos), "roll": list(d["roll"]), "turned": False})
    dl = _roll_json()
    if not legal_moves(pos, dl):
        nxt = _switch(pos)
        return jsonify({"pos": pos_to_json(nxt), "roll": None, "turned": True})
    return jsonify({"pos": pos_to_json(pos), "roll": list(dl), "turned": False})


@app.post("/api/agent_move")
def agent_move():
    d = request.json
    pos = pos_from_json(d["pos"])
    if not d.get("roll"):
        return jsonify({"error": "кости не брошены"}), 400
    nxt, kind, steps = _agent_choice(pos, tuple(d["roll"]))
    if nxt is None:
        return jsonify({"pos": pos_to_json(_switch(pos)), "moved": False, "agent": "none", "steps": []})
    return jsonify({"pos": pos_to_json(nxt), "moved": True, "agent": kind, "steps": steps})


@app.post("/api/meta")
def meta():
    global _loaded
    if NET is None and not _loaded:
        _loaded = _load_net()
    return jsonify(_META_BASE)