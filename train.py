#!/usr/bin/env python3
"""Запуск обучения агента нард (self-play + TD(λ)) с мониторингом.

Ключевые фичи для практики:
- **продолжение обучения**: `--resume` грузит последний чекпоинт и продолжает
  TensorBoard с реального шага (кривые не начинаются заново);
- **отдельный run на каждый запуск**: по умолчанию логдир = `<tblog>/<дата_время>`,
  так что разные запуски не смешиваются и TB показывает ровно один прогон;
- **чекпоинты** каждые `--save-every` + `net_final.pt`.

Использование:
    uv run python train.py --epochs 1000 --tblog runs/narды          # новый прогон
    uv run python train.py --resume --epochs 500 --tblog runs/narды  # продолжить
    uv run tensorboard --logdir runs/narды
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime

import torch

from core.board import Position  # noqa: F401  (для читаемости реплея)
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


def _last_ckpt(ckpt_dir: str) -> str | None:
    """Путь к чекпоинту с максимальным номером эпохи, либо None."""
    if not os.path.isdir(ckpt_dir):
        return None
    cands = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("net_") and f.endswith(".pt") and f != "net_final.pt":
            try:
                cands.append((int(f[4:-3]), os.path.join(ckpt_dir, f)))
            except ValueError:
                pass
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


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
    ap.add_argument("--resume", action="store_true", help="грузить последний чекпоинт и продолжить")
    ap.add_argument("--run-tag", default=None, help="метка запуска (по умолчанию дата_время)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    enc = Encoder()
    net = make_value_net(enc.dim(), hidden=args.hidden)
    net.train()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_step = 0
    if args.resume:
        last = _last_ckpt(args.ckpt_dir)
        if last is None:
            print("Нет чекпоинта — стартую с нуля.", flush=True)
        else:
            net.load_state_dict(torch.load(last, map_location="cpu"))
            start_step = int(os.path.basename(last)[4:-3])
            print(f"Resume: {last} (продолжаю с шага {start_step})", flush=True)

    # уникальный run-директорий на запуск, чтобы TB не мешал запуски
    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(args.tblog, tag)
    writer = SummaryWriter(logdir) if _HAS_TB else None
    repl_dir = os.path.join(logdir, "replays")

    print(f"TB: {'yes' if writer else 'no'} | logdir: {logdir}", flush=True)
    last_replay = time.time()

    W = 100
    wins, lens, rewards, losses = [], [], [], []
    for epoch in range(1, args.epochs + 1):
        step = start_step + epoch
        rng = random.Random(args.seed + step)
        traj, winner = play_one_game(net, enc, rng=rng, max_steps=args.max_steps)
        loss = train_episode(net, enc, traj, winner, lam=args.lam, lr=args.lr)

        win = 1.0 if winner == "white" else 0.0
        reward = 1.0 if winner == "white" else -1.0
        with torch.no_grad():
            ss = traj[:: max(1, len(traj) // 50)]
            Xs = torch.stack([torch.tensor(enc.encode(p), dtype=torch.float32) for p in ss])
            mv = float(net.value(Xs).mean())

        losses.append(float(loss)); wins.append(win); lens.append(len(traj)); rewards.append(reward)
        if len(wins) > W:
            losses.pop(0); wins.pop(0); lens.pop(0); rewards.pop(0)

        log = {
            "loss": float(loss),
            "loss_ema": _mean(losses),
            "winrate": _mean(wins),
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
                writer.add_scalar(f"train/{k}", v, step)
        else:
            if epoch % 20 == 0 or epoch == 1:
                print(f"step {step} | loss {float(loss):.4f} | wr {_mean(wins):.2f} | len {len(traj):3d}", flush=True)

        if time.time() - last_replay >= args.replay_every_min * 60:
            last_replay = time.time()
            print("[replay]", save_replay(traj, winner, repl_dir, step), flush=True)

        if step % args.save_every == 0:
            p = os.path.join(args.ckpt_dir, f"net_{step}.pt")
            torch.save(net.state_dict(), p)
            print(f"[ckpt] {p}", flush=True)

    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, "net_final.pt"))
    print("Готово. TB:", logdir, "| веса:", os.path.join(args.ckpt_dir, "net_final.pt"))


if __name__ == "__main__":
    main()