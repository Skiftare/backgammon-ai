#!/usr/bin/env python3
"""Запуск обучения агента нард (self-play + TD(λ)) с авто-мониторингом.

Одной командой запускает обучение, само поднимает TensorBoard на заданном
host:port, и грузит видеокарту (CUDA), если она есть. Параллелизм: на GPU —
batch по партиям через train_batch; на CPU (без GPU) — параллельная генерация
партий через --workers процессов.

Использование:
    uv run python train.py --epochs 5000 --host 0.0.0.0 --device cuda
    uv run python train.py --epochs 2000 --device cpu --workers 4
    uv run python train.py --resume --epochs 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import torch

from core.features import Encoder
from model.net import make_value_net
from training.selfplay import play_many_games, play_parallel
from training.td import train_batch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _last_ckpt(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    cands = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("net_") and f.endswith(".pt"):
            if f == "net_final.pt":
                continue
            try:
                cands.append((int(f[4:-3]), os.path.join(ckpt_dir, f)))
            except ValueError:
                pass
    if not cands:
        # нет номерных — попробуем net_final.pt (режим «доучи ещё N раундов»)
        fin = os.path.join(ckpt_dir, "net_final.pt")
        if os.path.exists(fin):
            return fin
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


def _port_busy(host: str, port: int) -> bool:
    """Проверить, отвечает ли уже что-то на host:port (TB уже запущен)."""
    import socket

    check_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((check_host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _start_tensorboard(logdir: str, host: str, port: int):
    """Авто-запуск tensorboard на РОДИТЕЛЬСКИЙ logdir (видит все run'ы).

    Если на порте уже крутится TB — не дублируем, просто сообщаем.
    """
    print(f"[tb] http://{host}:{port}  (logdir: {os.path.abspath(logdir)})", flush=True)
    if not _HAS_TB:
        print("[tb] tensorboard недоступен — лог в консоль", flush=True)
        return
    if _port_busy(host, port):
        print("[tb] уже запущен на этом порту — использую его", flush=True)
        return
    try:
        flags = [sys.executable, "-m", "tensorboard.main", "--logdir", logdir,
                 "--host", host, "--port", str(port)]
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(flags, creationflags=creation,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[tb] TensorBoard запущен (фоновый процесс)", flush=True)
    except Exception as e:
        print(f"[tb] не удалось запустить tensorboard: {e}", flush=True)


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
    ap.add_argument("--batch", type=int, default=1,
                    help="партий на один шаг обучения (GPU-параллелизм)")
    ap.add_argument("--workers", type=int, default=0,
                    help="процессы для генерации партий на CPU (0=в потоке)")
    ap.add_argument("--device", default=None,
                    help="auto/cpu/cuda (по умолчанию cuda, если доступна)")
    ap.add_argument("--host", default="127.0.0.1", help="адрес привязки TB (0.0.0.0 = сеть)")
    ap.add_argument("--tb-port", type=int, default=6006, help="порт TensorBoard")
    ap.add_argument("--resume", action="store_true", help="грузить последний чекпоинт")
    ap.add_argument("--run-tag", default=None, help="метка запуска (по умолчанию дата_время)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto" or args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dev] device={device} | cuda={torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'})", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    enc = Encoder()
    net = make_value_net(enc.dim(), hidden=args.hidden)
    net.train()
    net.to(device)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_step = 0
    if args.resume:
        last = _last_ckpt(args.ckpt_dir)
        if last is None:
            print("Нет чекпоинта — стартую с нуля.", flush=True)
        else:
            fin = os.path.basename(last)
            net.load_state_dict(torch.load(last, map_location="cpu"))
            start_step = 0 if fin == "net_final.pt" else int(fin[4:-3])
            print(f"Resume: {last} (продолжаю с шага {start_step})", flush=True)

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(args.tblog, tag)
    writer = SummaryWriter(logdir) if _HAS_TB else None
    repl_dir = os.path.join(logdir, "replays")

    _start_tensorboard(args.tblog, args.host, args.tb_port)
    print(f"[run] logdir={logdir} | ckpt={args.ckpt_dir}", flush=True)

    last_replay = time.time()
    W = 100
    wins, lens, rewards, losses = [], [], [], []
    n_played = 0

    for epoch in range(1, args.epochs + 1):
        step = start_step + epoch
        batch = max(1, args.batch)

        if device == "cuda" or args.workers == 0:
            games = play_many_games(net, enc, batch, seed_base=args.seed * 1000 + step,
                                    max_steps=args.max_steps)
        else:
            games = play_parallel(net, enc, batch, workers=args.workers,
                                  seed_base=args.seed * 1000 + step, max_steps=args.max_steps)
        n_played += len(games)

        loss = train_batch(net, enc, games, lam=args.lam, lr=args.lr, device=device)

        winners = [g[1] for g in games]
        win = sum(1.0 if w == "white" else 0.0 for w in winners) / len(winners)
        lens_this = [len(g[0]) for g in games]
        with torch.no_grad():
            mv = 0.0
            cnt = 0
            for traj, _ in games:
                ss = traj[:: max(1, len(traj) // 50)]
                Xs = torch.stack([torch.tensor(enc.encode(p), dtype=torch.float32) for p in ss]).to(device)
                mv += float(net.value(Xs).mean())
                cnt += 1
            mv = mv / max(1, cnt)

        losses.append(float(loss)); wins.append(win); lens.extend(lens_this); rewards.append(win * 2 - 1)
        if len(wins) > W:
            losses.pop(0); wins.pop(0); rewards.pop(0)
        lens = lens[-W:]

        log = {
            "loss": float(loss),
            "loss_ema": _mean(losses),
            "winrate": _mean(wins),
            "mean_value": mv,
            "game_len": _mean(lens_this),
            "reward_avg": _mean(rewards),
            "win_white": win,
            "win_black": 1.0 - win,
            "games_total": n_played,
        }
        if writer is not None:
            for k, v in log.items():
                writer.add_scalar(f"train/{k}", v, step)
        else:
            if epoch % 20 == 0 or epoch == 1:
                print(f"step {step} | loss {float(loss):.4f} | wr {_mean(wins):.2f} | len {int(_mean(lens_this)):3d}", flush=True)

        if time.time() - last_replay >= args.replay_every_min * 60:
            last_replay = time.time()
            for traj, winner in games[:1]:
                print("[replay]", save_replay(traj, winner, repl_dir, step), flush=True)

        if step % args.save_every == 0:
            p = os.path.join(args.ckpt_dir, f"net_{step}.pt")
            torch.save(net.state_dict(), p)
            print(f"[ckpt] {p}", flush=True)

    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, "net_final.pt"))
    print("Готово. TB:", logdir, "| веса:", os.path.join(args.ckpt_dir, "net_final.pt"), flush=True)


if __name__ == "__main__":
    main()