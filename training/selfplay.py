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
    """Воркер для ProcessPoolExecutor: грузит CPU-копию сети и генерит n партий
    с батченной оценкой кандидатов (simulate_games_batched) — векторно, быстро.
    """
    import io

    import torch

    from model.net import make_value_net

    torch.set_num_threads(1)  # см. _play_worker_memo: борьба с oversubscription

    net = make_value_net(cfg["in_dim"], hidden=cfg["hidden"], layers=cfg["layers"])
    buf = io.BytesIO(cfg["state_bytes"])
    net.load_state_dict(torch.load(buf, map_location="cpu"))
    net.eval()
    enc = Encoder()
    return simulate_games_batched(
        net, enc, cfg["n"], device="cpu",
        seed_base=cfg["seed_base"], max_steps=cfg["max_steps"], eps=cfg["eps"],
    )


def build_worker_cfgs(net: ValueNet, batch: int, workers: int, seed_base: int,
                      max_steps: int = 1000, eps: float = 0.0) -> list[dict]:
    """Собрать конфиги для параллельной генерации `batch` партий на `workers` ядер."""
    import io

    import torch

    # hidden = ширина первого скрытого слоя, layers = число скрытых слоёв
    lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    hidden = lin[0].out_features if lin else 512
    layers = len(lin) - 1 if lin else 2
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    in_dim = next(m.in_features for m in net.modules() if isinstance(m, torch.nn.Linear))

    per = max(1, batch // workers)
    cfgs = []
    for w in range(workers):
        n = per if w < workers - 1 else batch - per * (workers - 1)
        cfgs.append({
            "in_dim": in_dim, "hidden": hidden, "layers": layers, "n": n,
            "seed_base": seed_base + w * 7919,
            "max_steps": max_steps, "eps": eps, "state_bytes": buf.getvalue(),
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


def simulate_games_batched(net: ValueNet, encoder: Encoder, n: int, device: str = "cpu",
                           seed_base: int = 0, max_steps: int = 1000, eps: float = 0.0):
    """Генерация n партий ОДНОВРЕМЕННО, с батченной оценкой кандидатов на GPU.

    Каждый плюс: для КАЖДОЙ активной партии перебираем легальные ходы (движок на CPU),
    кандидаты ВСЕХ партий кодируются в один большой тензор и прогоняются одним forward —
    это по-настоящему грузит видеокарту (вместо K×маленьких вызовов). ε-greedy
    (`eps>0`) даёт exploration и снимает «травоядность» локального жадного режима.
    """
    import torch

    from core.game import legal_moves, apply_move

    # состояние партии: позиция, траектория, активна?
    positions = [Position.initial() for _ in range(n)]
    trajs: list[list[Position]] = [[p] for p in positions]
    alive = [True] * n
    rg = random.Random(seed_base)

    whome = lambda p: p.turn == "white"

    for _ in range(max_steps):
        active = [i for i in range(n) if alive[i]]
        if not active:
            break

        # 1) бросок и легальные кандидаты для каждого активного
        rows: list[list[Position]] = []   # rows[i] = кандидаты игры active[i]
        rolls: list = []
        for i in active:
            a, b = rg.randint(1, 6), rg.randint(1, 6)
            roll = (a, a, a, a) if a == b else (a, b)
            moves = legal_moves(positions[i], roll)
            rolls.append(roll)
            if moves:
                rows.append([apply_move(positions[i], m) for m in moves])
            else:
                # хода нет — фишки не трогаем, меняем сторону
                rows.append([None])  # маркер «пасс»
            # проверим терминал прямо тут (не ждать след. цикл)
            p = positions[i]
            if p.home_white == 15 or p.home_black == 15:
                alive[i] = False

        # 2) один батч-forward по ВСЕМ кандидатам
        flat = []
        for r in rows:
            flat.extend(x for x in r if x is not None)
        if flat:
            X = torch.tensor(encoder.encode_batch(flat), dtype=torch.float32, device=device)
            with torch.no_grad():
                V = net.value(X)
        else:
            V = torch.tensor([], dtype=torch.float32)

        # 3) применяем выбор хода к каждой партии
        off = 0
        for k, i in enumerate(active):
            cands = rows[k]
            if len(cands) == 0:
                alive[i] = False
                continue
            if len(cands) == 1 and cands[0] is None:
                # пасс: меняем сторону, фишки не трогаем
                positions[i] = Position(
                    points=positions[i].points,
                    bar_white=positions[i].bar_white,
                    bar_black=positions[i].bar_black,
                    home_white=positions[i].home_white,
                    home_black=positions[i].home_black,
                    turn="black" if positions[i].turn == "white" else "white",
                )
                trajs[i].append(positions[i])
                continue
            # value пачки кандидатов этой партии
            sub = V[off:off + len(cands)]
            off += len(cands)
            if eps and rg.random() < eps:
                pick = rg.randrange(len(cands))
            else:
                pick = int(torch.argmax(sub)) if whome(positions[i]) else int(torch.argmin(sub))
            positions[i] = cands[pick]
            trajs[i].append(positions[i])
            p = positions[i]
            if p.home_white == 15 or p.home_black == 15:
                alive[i] = False

        if not any(alive):
            break

    games = []
    for i, t in enumerate(trajs):
        winner = "white" if t[-1].home_white >= t[-1].home_black else "black"
        games.append((list(t), winner))
    return games