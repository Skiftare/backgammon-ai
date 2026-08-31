"""Self-play: генерация партии с greedy-политикой по value-сети.

Обучение value-сети (TD) идёт по траекториям, которые генерят эти партии.
Каждый ход: бросок костей (честный), список легальных ходов из движка, выбор
жадным по value позиции после хода.

Победа: у игрока вынесены все 15 фишек (в нашей нотации home_white/home_black == 15).
Если ходов нет (ход пропадает) — просто смена стороны без передвижений.
"""

from __future__ import annotations

import random

import torch

from core.board import TOTAL_CHECKERS, Position
from core.features import Encoder
from core.game import apply_move, legal_moves
from model.net import ValueNet


def _other(turn: str) -> str:
    return "black" if turn == "white" else "white"


def choose_greedy(pos: Position, roll, net: ValueNet, encoder: Encoder) -> Position:
    """Применить лучший ход greedy по value (для текущего игрока).

    Все кандидаты оцениваются ОДНИМ батч-forward'ом (net.value(X) на K кандидатах)
    — это догружает GPU/CPU вместо K отдельных вызовов маленьких тензоров.
    """
    moves = legal_moves(pos, roll)
    if not moves:
        # хода нет — смена стороны (фишки не трогаем)
        return Position(
            points=pos.points,
            bar_white=pos.bar_white,
            bar_black=pos.bar_black,
            home_white=pos.home_white,
            home_black=pos.home_black,
            turn=_other(pos.turn),
        )

    nxts = [apply_move(pos, m) for m in moves]
    dev = next(net.parameters()).device
    X = torch.stack([torch.tensor(encoder.encode(p), dtype=torch.float32) for p in nxts]).to(dev)
    with torch.no_grad():
        V = net.value(X)  # (K,) value «для ходящего» следующей позиции
    # белый хочет максимум value, чёрный — минимум (как в трэин-логике td.py)
    idx = int(torch.argmax(V)) if pos.turn == "white" else int(torch.argmin(V))
    return nxts[idx]


def play_one_game(net: ValueNet, encoder: Encoder, rng=None, max_steps: int = 1000):
    """Играет одну партию greedy-политикой.

    Возвращает (траектория позиций, победитель_цвет) где победитель = 'white'|'black'.
    На исчерпание max_steps возвращается лидер по числу вынесенных (безопасность).
    """
    if rng is None:
        rng = random.Random()
    pos = Position.initial()
    traj = [pos]
    for _ in range(max_steps):
        # победа?
        if pos.home_white == TOTAL_CHECKERS:
            return traj, "white"
        if pos.home_black == TOTAL_CHECKERS:
            return traj, "black"

        a = rng.randint(1, 6)
        b = rng.randint(1, 6)
        roll = (a, a, a, a) if a == b else (a, b)

        nxt = choose_greedy(pos, roll, net, encoder)
        traj.append(nxt)
        pos = nxt

    winner = "white" if pos.home_white >= pos.home_black else "black"
    return traj, winner


def play_many_games(net: ValueNet, encoder: Encoder, n: int, seed_base: int = 0, max_steps: int = 1000):
    """Последовательно сыграть n партий (для GPU/одиночного потока)."""
    games = []
    for i in range(n):
        rng = random.Random(seed_base + i)
        traj, winner = play_one_game(net, encoder, rng=rng, max_steps=max_steps)
        games.append((list(traj), winner))
    return games


def _play_worker(cfg: dict):
    """Воркер для ProcessPoolExecutor (Windows-spawn безопасен: всё сериализуемо).

    Грузит CPU-копию сети из байтов и генерит cfg['n'] партий.
    """
    import io

    import torch

    from model.net import make_value_net

    net = make_value_net(cfg["in_dim"], hidden=cfg["hidden"])
    buf = io.BytesIO(cfg["state_bytes"])
    net.load_state_dict(torch.load(buf, map_location="cpu"))
    net.eval()
    enc = Encoder()
    return play_many_games(net, enc, cfg["n"], seed_base=cfg["seed_base"], max_steps=cfg["max_steps"])


def build_worker_cfgs(net: ValueNet, batch: int, workers: int, seed_base: int,
                      max_steps: int = 1000) -> list[dict]:
    """Собрать конфиги для параллельной генерации `batch` партий на `workers` ядер."""
    import io

    import torch

    # hidden = ширина первого скрытого слоя
    hidden = next(m.out_features for m in net.modules() if isinstance(m, torch.nn.Linear))
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    in_dim = next(m.in_features for m in net.modules() if isinstance(m, torch.nn.Linear))

    per = max(1, batch // workers)
    cfgs = []
    for w in range(workers):
        n = per if w < workers - 1 else batch - per * (workers - 1)
        cfgs.append({
            "in_dim": in_dim, "hidden": hidden, "n": n,
            "seed_base": seed_base + w * 7919,
            "max_steps": max_steps, "state_bytes": buf.getvalue(),
        })
    return cfgs


def play_parallel(net: ValueNet, encoder: Encoder, n: int, workers: int,
                  seed_base: int = 0, max_steps: int = 1000):
    """Параллельная генерация партий через процессы (для CPU-машины без GPU).

    Сериализует веса в байты и раскидывает по воркерам; возвращает список
    кортежей (траектория, победитель) — пригодный для train_batch.
    """
    import io

    import torch
    from concurrent.futures import ProcessPoolExecutor

    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)

    games: list[tuple[list, str]] = []
    # hidden = ширина первого скрытого слоя
    hidden = None
    for mod in net.modules():
        if isinstance(mod, torch.nn.Linear):
            hidden = mod.out_features
            break
    cfg_base = {
        "in_dim": encoder.dim(),
        "hidden": hidden,
        "max_steps": max_steps,
        "state_bytes": buf.getvalue(),
    }

    per = max(1, n // workers)
    tasks = []
    for w in range(workers):
        cfg = dict(cfg_base, n=per, seed_base=seed_base + w * 7919)
        tasks.append(cfg)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_play_worker, tasks):
            games.extend(res)
    return games[:n]