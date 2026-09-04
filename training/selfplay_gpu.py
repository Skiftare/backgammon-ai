"""GPU-рейди self-play: pad-to-K батч-выбор + мемоизированный движок ходов.

Почему такая архитектура (всё обосновано в `_verify_gpu_offload.py`, 18/18 зелёные;
профиль показывал ~99% CPU в Python-движке, GPU видел только крошечный TD-шаг):

- **CPU-узкое место** — `legal_moves` (DFS) + `_apply_step`. На дублях DFS-дерево
  взрывается пермутациями одинаковых костей: 15 509 узлов -> 126 уникальных
  поддеревьев. Мемоизация по `(compress_state, мультимножество оставшихся костей)`
  даёт ~123x экономию на дубле, ~3.2x end-to-end (CPU), набор ходов идентичен.
- **GPU-узкое место** — не было: сеть видела только TD-шаг. Здесь выбор хода
  делается ОДНИМ батч-forward'ом по ВСЕМ кандидатам всех партий + pad-to-K
  тензорный argmax (одна синхронизация на ply) вместо K×маленьких вызовов.
- **0/1-кеш НЕ взят**: формально неверен (см. verify C). Для value нужен ТОЧНЫЙ
  ключ (`pack_exact`, 141 бит, биекция); он окупается ТОЛЬКО при лукахеде
  (dup-rate пласта 44-49% — проверено в `_measure_tt.py`), в 1-пльном greedy —
  0.01% хитов, поэтому без лукахеда value-кеш не строим.

Семантика: при eps=0 траектории побитово идентичны старому `simulate_games_batched`
(проверено), при eps>0 greedy-политика эквивалентна.
"""

from __future__ import annotations

import random

import torch

from core.board import N_POINTS, Position
from core.features import Encoder
from core.game import apply_move, legal_moves
from core.moves import Move

# ---------------------------------------------------------------- ключи ----

def compress_state(pos: Position) -> int:
    """Сжатый ключ состояния для мемо движка: own∈{0..4}×opp∈{0,1,2} на точку
    (15 состояний = 4 бита) + бар ходящего (4 бита) = 100 бит.

    Достаточность (verify B, 12 600 сравнений): легальность хода зависит от
      - own >= k (можно снять k костей с точки при дубле; k<=4 => cap 4),
      - opp <= 1 (пусто/блот-хват) vs >=2 (закрыто),
      - бар ходящего (до 4 входов).
    0/1 ломается на дубле со стопкой >=3.
    """
    s = 0
    for i in range(N_POINTS):
        v = pos.points[i]
        sv = v if pos.turn == "white" else -v
        own = 0 if sv <= 0 else min(int(sv), 4)
        opp = 0 if sv >= 0 else (1 if sv == -1 else 2)
        s = (s << 4) | (own * 3 + opp)
    bar_me = pos.bar_white if pos.turn == "white" else pos.bar_black
    return (s << 4) | bar_me


def pack_exact(pos: Position) -> int:
    """Биективный ключ позиции (141 бит, один Python int, без коллизий).

    НЕ зеркалим: энкодер при смене ходящего инвертирует только знак, индексы не
    отражаются (verify A). Точная канонизация = сама позиция как есть.
    """
    k = 0
    for v in pos.points:
        k = (k << 5) | (v + 15)
    k = (k << 5) | pos.bar_white
    k = (k << 5) | pos.bar_black
    k = (k << 5) | pos.home_white
    k = (k << 5) | pos.home_black
    return (k << 1) | (0 if pos.turn == "white" else 1)


# ------------------------------------------------------------- движок ------

def legal_moves_memo(pos: Position, roll, memo: dict | None = None) -> list[Move]:
    """legal_moves с мемоизацией под-DFS: ключ (compress_state, мультимножество
    оставшихся костей). Дубли перестают плодить пермутации: поддерево
    (состояние, {3 кости}) считается один раз (раньше — ~4!/k! раз).

    Выход: канонический (отсортированный) список ходов максимальной длины.
    Набор ходов ИДЕНТИЧЕН legal_moves (verify F2); порядок — сортировка.
    """
    from core.game import _apply_step, _single_moves

    memo = memo if memo is not None else {}
    steps_all = list(roll)

    def dfs(pos_: Position, rem: list[int]):
        rem_t = tuple(sorted(rem))
        mk = (compress_state(pos_), rem_t)
        hit = memo.get(mk)
        if hit is not None:
            return hit
        if not rem:
            res: list = [()]
        else:
            any_ok = False
            res = []
            for i, st in enumerate(rem):
                for mv in _single_moves(pos_, st):
                    any_ok = True
                    nxt = _apply_step(pos_, mv)
                    new_rem = rem[:i] + rem[i + 1:]
                    for tail in dfs(nxt, new_rem):
                        res.append((mv,) + tail)
            if not any_ok:
                res = [()]  # мёртвый узел
        memo[mk] = res
        return res

    paths = [p for p in dfs(pos, steps_all) if p]
    uniq = sorted(set(paths), key=len, reverse=True)
    if not uniq:
        return []
    maxlen = len(uniq[0])
    best = [p for p in uniq if len(p) == maxlen]
    return [Move(steps=p) for p in best]


# ------------------------------------------------------------- энкодер ------

def encode_batch_gpu(positions, device="cpu", enc: Encoder | None = None) -> torch.Tensor:
    """Энкодинг прямо тензорами (без numpy): с CPU передаётся только (N,24) int16,
    разворачивание в 293-d пороговых слоёв — на GPU. Совпадает с
    Encoder.encode_batch (verify E, max|Δ| ~7e-9). ~12x меньше H2D-трафика.
    """
    enc = enc or Encoder()
    M = enc.max_on_point
    N = len(positions)
    pts = torch.tensor([p.points for p in positions], dtype=torch.int16, device=device)
    turn_w = torch.tensor([p.turn == "white" for p in positions], dtype=torch.bool, device=device)
    v = torch.where(turn_w[:, None], pts, -pts).to(torch.float32)
    own = torch.clamp(v, min=0.0)
    opp = torch.clamp(-v, min=0.0)

    def side(c: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(c, max=float(M))
        mask = c[:, :, None] >= torch.arange(1, M + 1, device=device)  # (N,24,M)
        return mask.reshape(N, -1).to(torch.float32)

    bars_w = torch.tensor([p.bar_white for p in positions], dtype=torch.float32, device=device)
    bars_b = torch.tensor([p.bar_black for p in positions], dtype=torch.float32, device=device)
    homes_w = torch.tensor([p.home_white for p in positions], dtype=torch.float32, device=device)
    homes_b = torch.tensor([p.home_black for p in positions], dtype=torch.float32, device=device)
    tail = torch.stack([
        torch.where(turn_w, bars_w, bars_b) / 15.0,
        torch.where(turn_w, homes_w, homes_b) / 15.0,
        torch.where(turn_w, bars_b, bars_w) / 15.0,
        torch.where(turn_w, homes_b, homes_w) / 15.0,
    ], dim=1)
    if enc.include_turn:
        tail = torch.cat([tail, turn_w.to(torch.float32)[:, None]], dim=1)
    return torch.cat([side(own), side(opp), tail], dim=1)


# ------------------------------------------------------- основной цикл ------

def simulate_games_batched_v2(net, n: int = 64, device: str = "cpu",
                              seed_base: int = 0, max_steps: int = 1000, eps: float = 0.0,
                              encoder: Encoder | None = None, memo_engine: bool = True,
                              fast_encode: bool = False, memo: dict | None = None,
                              select: str = "old"):
    """Аналог simulate_games_batched: генерация n партий с батченной оценкой
    кандидатов. Все device-агностичны; на сервере тот же код с device='cuda'.

    select: 'old' = как в текущем коде (белый argmax V кандидата, чёрный argmin).
            'fixed' = оба argmin V кандидата (математически обоснованно: V кандидата
            считается с перспективы ХОДЯЩЕГО в кандидате, т.е. после хода это шанс
            СОПЕРНИКА; свой шанс = -V, максимизировать его = argmin V).
            Эксперимент по подтверждению 'fixed' полезным — следующий шаг
            (решающий тест `_verify_sign_train`); по умолчанию — 'old', чтобы
            НЕ менять поведение существующего train.

    Лукахед с транспозиционной таблицей (value-кеш по точному pack_exact) — СЛЕДУЮЩИЙ
    шаг: замер dup-rate пласта показал 44-49% (т.е. таблица окупится при depth>=2),
    но реализация требует батчеванного DFS — не в этом коммите.
    """
    enc = encoder or Encoder()
    positions = [Position.initial() for _ in range(n)]
    trajs: list[list[Position]] = [[p] for p in positions]
    alive = [True] * n
    rg = random.Random(seed_base)

    if memo is None:
        memo = {}

    def lm(pos, roll):
        if memo_engine:
            return legal_moves_memo(pos, roll, memo)
        return legal_moves(pos, roll)

    def other(t): return "black" if t == "white" else "white"

    for _ in range(max_steps):
        active = [i for i in range(n) if alive[i]]
        if not active:
            break

        # 1) бросок + легальные кандидаты для каждого активного
        rows: list[list] = []
        for i in active:
            a, b = rg.randint(1, 6), rg.randint(1, 6)
            roll = (a, a, a, a) if a == b else (a, b)
            moves = lm(positions[i], roll)
            rows.append([apply_move(positions[i], m) for m in moves] if moves else [None])
            p = positions[i]
            if p.home_white == 15 or p.home_black == 15:
                alive[i] = False

        # 2) один батч-forward по ВСЕМ кандидатам
        G = len(active)
        if G == 0:
            break
        K = max(len(r) for r in rows)
        flat: list[Position] = []
        for r in rows:
            flat.extend(x for x in r if x is not None)
        if flat:
            X = ((enc.encode_batch(flat) if not fast_encode else
                  encode_batch_gpu(flat, device=device, enc=enc).cpu().numpy()))
            Xt = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                V = net.value(Xt)
        else:
            V = torch.tensor([], dtype=torch.float32, device=device)

        # --- pad-to-K матрица выбора ---
        # select='old': белый +V (argmax V), чёрный -V (argmin V) — поведение
        #               текущего кода (для обратной совместимости).
        # select='fixed': оба -V -> оба argmin V кандидата (V с перспективы
        #               соперника; свой шанс максимизируется минимизацией V).
        M = torch.full((G, K), float("-inf"), dtype=torch.float32, device=device)
        off = 0
        for k, i in enumerate(active):
            r = rows[k]
            if len(r) == 1 and r[0] is None:
                continue
            L = len(r)
            if select == "fixed":
                sg = -1.0
            else:
                sg = 1.0 if positions[i].turn == "white" else -1.0
            M[k, :L] = sg * V[off:off + L]
            off += L
        picks = torch.argmax(M, dim=1).tolist()

        # 3) применяем
        for k, i in enumerate(active):
            cands = rows[k]
            if len(cands) == 1 and cands[0] is None:
                positions[i] = Position(points=positions[i].points,
                                        bar_white=positions[i].bar_white,
                                        bar_black=positions[i].bar_black,
                                        home_white=positions[i].home_white,
                                        home_black=positions[i].home_black,
                                        turn=other(positions[i].turn))
            else:
                if eps and rg.random() < eps:
                    pick = rg.randrange(len(cands))
                else:
                    pick = picks[k]
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


# --------------------------------------------------- параллельная генерация

def _play_worker_memo(cfg: dict):
    """Воркер для ProcessPoolExecutor: CPU-копия сети + МЕМО-движок.

    В отличие от `selfplay._play_worker` (старый legal_moves, 1 ядро/процесс),
    здесь `simulate_games_batched_v2` использует мемоизированный DFS — на дублях
    ~123x меньше DFS-узлов, поэтому каждый процесс генерёт сильно быстрее.
    Так все ядра работают, а главный процесс на GPU только кодирует и учит.
    """
    import io

    import torch

    from model.net import make_value_net

    net = make_value_net(cfg["in_dim"], hidden=cfg["hidden"], layers=cfg["layers"])
    buf = io.BytesIO(cfg["state_bytes"])
    net.load_state_dict(torch.load(buf, map_location="cpu"))
    net.eval()
    enc = Encoder()
    memo: dict = {}
    return simulate_games_batched_v2(
        net, n=cfg["n"], device="cpu", seed_base=cfg["seed_base"],
        max_steps=cfg["max_steps"], eps=cfg["eps"], encoder=enc,
        memo_engine=True, memo=memo, select=cfg.get("select", "old"),
    )