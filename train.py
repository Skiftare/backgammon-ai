#!/usr/bin/env python3
"""Запуск обучения агента нард (self-play + TD(λ)) с мониторингом.

Использование:
    uv run python train.py --epochs 1000 --tblog runs/narды

Цикл:
- каждая «эпоха» = одна партия self-play (greedy по value-сети);
- после партии — TD(λ)-обновление (training.td.train_episode);
- метрики в TensorBoard:
    * loss (TD-error²) + EMA loss      — главная кривая сходимости
    * winrate (скользящее окно)        — accuracy агента в self-play
    * mean_value                       — калибровка value (средний прогноз)
    * game_len                         — длина партии (короче → сильнее)
    * reward (средний)                 — итог партии (+1/0/-1)
    * win_white / win_black            — дисбаланс сторон
- чекпоинты весов, реплей-last (JSON) раз в N минут (дефолт 5).

TensorBoard: `uv run tensorboard --logdir runs/narды`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime

import torch

from core.features import Encoder
from model.net import make_value_net
from training.selfplay import play_one_game
from training.td import train_episode

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def save_replay(traj, winner, out_dir: str, step: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    path = os.path.join(out_dir, f"replay_{step}_{ts}.json")
    payload = {
        "step": step, "winner": winner, "ts": ts, "n_positions": len(traj),
        "positions": [list(p.points) + [p.bar_white, p.bar_black, p.home_white, p.home_black,
                                        (0 if p.turn == "white" else 1)] for p in traj],
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

    print(f"TB: {'yes' if writer else 'no (tensorboard не установлен)'} | log: {args.tblog}", flush=True)
    last_replay = time.time()

    # накопители метрик (окно для скользящих)
    W = 100
    wins, lens, rewards, losses = [], [], [], []
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        traj, winner = play_one_game(net, enc, rng=rng, max_steps=args.max_steps)
        loss = train_episode(net, enc, traj, winner, lam=args.lam, lr=args.lr)

        # метрики партии
        win = 1.0 if winner == "white" else 0.0
        # reward: +1 если ходящий победил (для белого), иначе -1 (простая форма)
        reward = 1.0 if winner == "white" else -1.0
        # mean value сети по траектории (калибровка value-сети)
        with torch.no_grad():
            Xs = torch.stack([torch.tensor(enc.encode(p), dtype=torch.float32) for p in traj[:: max(1, len(traj) // 50)]])
            mv = float(net.value(Xs).mean())
        loss_v = float(loss)

        losses.append(loss_v); wins.append(win); lens.append(len(traj)); rewards.append(reward)
        if len(wins) > W:  # скользящее окно
            wins.pop(0); lens.pop(0); rewards.pop(0); losses.pop(0)

        log = {
            "loss": loss_v,
            "loss_ema": _mean(losses),
            "winrate": _mean(wins),           # accuracy self-play (окно W)
            "winrate_cum": (sum(wins) ) / len(wins),
            "mean_value": mv,
            "game_len": len(traj),
            "game_len_avg": _mean(lens),
            "reward": reward,
            "reward_avg": _mean(rewards),
            "win_white": win,
            "win_black": 1.0 - win,
        }
        if writer is not None:
            for k, v in log.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            # hi-метрики в одну таблицу (гистограммы/скаляры в графике)
            writer.add_scalar("train/EMA", log["loss_ema"], epoch)
        else:
            if epoch % 20 == 0 or epoch == 1:
                print(f"epoch {epoch:4d} | loss {loss_v:.4f} | wr {_mean(wins):.2f} | len {len(traj):3d} | mv {mv:.3f}", flush=True)

        # реплей раз в N минут
        if time.time() - last_replay >= args.replay_every_min * 60:
            last_replay = time.time()
            path = save_replay(traj, winner, repl_dir, epoch)
            print(f"[replay] {path}", flush=True)

        if epoch % args.save_every == 0:
            p = os.path.join(args.ckpt_dir, f"net_{epoch}.pt")
            torch.save(net.state_dict(), p)
            print(f"[ckpt] {p}", flush=True)

    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, "net_final.pt"))
    print("Готово. TB:", args.tblog, "| веса:", os.path.join(args.ckpt_dir, "net_final.pt"))


if __name__ == "__main__":
    main()