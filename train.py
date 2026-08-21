#!/usr/bin/env python3
"""Запуск обучения агента нард (self-play + TD(λ)) с мониторингом.

Использование:
    uv run python train.py --epochs 1000 --replay-every 10 --tblog runs/narды

Что делает:
- каждая «эпоха» = одна партия self-play (greedy-policy по value-сети);
- после партии — TD(λ) обновление (training.td.train_episode);
- каждые `--replay-every` минут выгружается реплей последней партии
  (в runs/<ts>/replay_<step>.json + текстовая расшифровка), чтобы смотреть,
  как агент улучшается;
- метрики в TensorBoard: loss/TD-error, длина партии, победитель, (позже) winrate;
- чекпоинт весов каждые `--save-every` итераций.

Для нужных пакетов: uv add torch (на целевой машине); tensorboard — по желанию
(если нет — метрики пишутся в CSV, TB-совместимо).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch

from core.board import Position
from core.features import Encoder
from model.net import make_value_net
from training.selfplay import play_one_game
from training.td import train_episode

# ── tensorboard (опционально) ──
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


def log_metrics(writer, tag, values: dict, step: int) -> None:
    if writer is not None:
        for k, v in values.items():
            writer.add_scalar(f"{tag}/{k}", v, step)


def save_replay(traj, winner, out_dir: str, step: int) -> str:
    """Сохранить партию в JSON (позиции, ходы нет — но траектория видна)."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    path = os.path.join(out_dir, f"replay_{step}_{ts}.json")
    payload = {
        "step": step,
        "winner": winner,
        "ts": ts,
        "n_positions": len(traj),
        "positions": [list(p.points) + [p.bar_white, p.bar_black, p.home_white, p.home_black, (0 if p.turn == "white" else 1)] for p in traj],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tblog", default="runs/narды")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--replay-every-min", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    enc = Encoder()
    net = make_value_net(enc.dim(), hidden=args.hidden)
    net.train()

    writer = SummaryWriter(args.tblog) if _HAS_TB else None
    os.makedirs(args.ckpt_dir, exist_ok=True)
    repl_dir = os.path.join(args.tblog, "replays")
    os.makedirs(repl_dir, exist_ok=True)

    print(f"TB: {'да' if writer else 'нет (нет tensorboard)'} | лог: {args.tblog}", flush=True)
    last_replay = time.time()
    losses = []
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        traj, winner = play_one_game(net, enc, rng=rng, max_steps=args.max_steps)
        loss = train_episode(net, enc, traj, winner, lam=args.lam, lr=args.lr)
        losses.append(loss)

        log_metrics(writer, "train", {
            "loss": loss,
            "game_len": len(traj),
            "win_white": 1.0 if winner == "white" else 0.0,
            "win_black": 1.0 if winner == "black" else 0.0,
            "ema_loss": sum(losses[-50:]) / min(len(losses), 50),
        }, epoch)

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | loss {loss:.4f} | len {len(traj):3d} | winner {winner}", flush=True)

        # реплей раз в N минут
        if time.time() - last_replay >= args.replay_every_min * 60:
            last_replay = time.time()
            path = save_replay(traj, winner, repl_dir, epoch)
            print(f"[replay] сохранён: {path}", flush=True)

        # чекпоинт
        if epoch % args.save_every == 0:
            p = os.path.join(args.ckpt_dir, f"net_{epoch}.pt")
            torch.save(net.state_dict(), p)
            print(f"[ckpt] {p}", flush=True)

    # финал
    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, "net_final.pt"))
    print("Готово. Тензорборд:", args.tblog, "| веса:", os.path.join(args.ckpt_dir, "net_final.pt"))


if __name__ == "__main__":
    main()