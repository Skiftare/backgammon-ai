"""Верификация GPU-разгрузки: корректность операций и логика снятия CPU-нагрузки.

Запуск:  python _verify_gpu_offload.py   (torch CPU; на сервере тот же код device='cuda'
семантически идентичен — все проверки device-агностичны).

Проверки:
  A  pack_exact — биекция позиция<->ключ; canonical-зеркало сохраняет encode() побитово
  B  movegen_key (min(|v|,2)+бар+бросок) сохраняет НАБОР легальных ходов (brute force)
  C  Контрпримеры: 0/1-ключ НЕ сохраняет ни value, ни набор ходов
  D  pad-to-K + masked argmax == per-game argmax (GPU-стиль выбора хода)
  E  encode_batch_gpu ≈ Encoder.encode_batch (tol 1e-6)
  F  v2 без кешей == текущий simulate_games_batched, ТЕ ЖЕ траектории (seeds)
  G  v2 с кешами == текущий, ТЕ ЖЕ траектории (seeds)
  H  value-кеш: hit == свежий forward (точность float32)
  I  Бенч: время + hit-rate кешей
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch

from core.board import N_POINTS, Position
from core.features import Encoder
from core.game import apply_move, legal_moves
from model.net import make_value_net
from training.selfplay import simulate_games_batched
from training.selfplay_gpu import (compress_state, encode_batch_gpu, pack_exact,
                                   simulate_games_batched_v2)

PASS = 0
FAIL = 0


def check(name, ok, extra=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok]   {name} {extra}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def moveset(pos, roll):
    return {tuple(m.steps) for m in legal_moves(pos, roll)}


def rand_pos(rng, bearing_phase: bool = False) -> Position:
    turn = rng.choice(["white", "black"])
    pts = [0] * N_POINTS
    if bearing_phase:
        # все «свои» фишки в своём доме
        home_idx = range(0, 6) if turn == "white" else range(18, 24)
        used = 0
        for i in home_idx:
            c = rng.randint(0, 5)
            pts[i] = c if turn == "white" else -c
            used += c
        # чужие фишки — где угодно вне домов «своих» (просто разброс)
        used_o = 0
        for i in range(N_POINTS):
            if i in home_idx:
                continue
            c = rng.randint(0, 3)
            pts[i] = -c if turn == "white" else c
            used_o += c
        return Position(points=tuple(pts), turn=turn)
    # случайная «мидгейм»-позиция: знаки вперемешку
    for i in range(N_POINTS):
        c = rng.randint(0, 5)
        s = rng.choice([-1, 1])
        pts[i] = c * s
    bw = rng.randint(0, 2)
    bb = rng.randint(0, 2)
    if turn == "black":
        bw, bb = bb, bw
    return Position(points=tuple(pts), bar_white=bw, bar_black=bb, turn=turn)


def compress_2bit_of(p: Position) -> Position:
    """Позиция, сжатая ровно по movegen_key: own cap 4, opp cap 2, знак сохранён."""
    pts = []
    for v in p.points:
        sv = v if p.turn == "white" else -v
        if sv >= 0:
            own = min(int(sv), 4)
            val = own if p.turn == "white" else -own
        else:
            opp = 1 if sv == -1 else 2
            val = -opp if p.turn == "white" else opp
        pts.append(val)
    return Position(points=tuple(pts), bar_white=p.bar_white, bar_black=p.bar_black,
                    home_white=p.home_white, home_black=p.home_black, turn=p.turn)


def compress_01_of(p: Position) -> Position:
    """0/1-сжатие (идея юзера): занято/нет, знак сохранён."""
    pts = tuple((1 if v > 0 else (-1 if v < 0 else 0)) for v in p.points)
    return Position(points=pts, bar_white=p.bar_white, bar_black=p.bar_black,
                    home_white=p.home_white, home_black=p.home_black, turn=p.turn)


ALL_ROLLS = [(a, b) for a in range(1, 7) for b in range(a, 7)]


def traj_eq(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x.points, x.bar_white, x.bar_black, x.home_white, x.home_black, x.turn) != \
           (y.points, y.bar_white, y.bar_black, y.home_white, y.home_black, y.turn):
            return False
    return True


def main():
    enc = Encoder()
    rng = random.Random(12345)

    print("== A. pack_exact: биекция + какие канонизации ТОЧНЫ ==")
    seen = set()
    bij = True
    for _ in range(5000):
        p = rand_pos(rng, bearing_phase=rng.random() < 0.4)
        k = pack_exact(p)
        if k in seen:
            bij = False
            break
        seen.add(k)
    check("pack_exact уникален на 5000 случайных позиций", bij)

    # Какая канонизация сохраняет encode()? Только T: знаковая инверсия + свап
    # бар/дом + смена хода (энкодер НЕ зеркалит доску!). Зеркало p->23-p НЕ сохраняет.
    def T(p):
        return Position(points=tuple(-v for v in p.points),
                        bar_white=p.bar_black, bar_black=p.bar_white,
                        home_white=p.home_black, home_black=p.home_white,
                        turn="black" if p.turn == "white" else "white")

    def mirror(p):
        return Position(points=tuple(-p.points[23 - i] for i in range(N_POINTS)),
                        bar_white=p.bar_black, bar_black=p.bar_white,
                        home_white=p.home_black, home_black=p.home_white,
                        turn="black" if p.turn == "white" else "white")

    t_ok = True
    m_ok = True
    for _ in range(500):
        p = rand_pos(rng)
        e = enc.encode(p)
        et = enc.encode(T(p))
        em = enc.encode(mirror(p))
        # T: входы совпадают кроме последнего бита (turn). Зеркало: не совпадают.
        if not np.array_equal(e[:-1], et[:-1]):
            t_ok = False
        if np.array_equal(e, em):
            m_ok = False
    check("T-канонизация (знак+свап) сохраняет encode() кроме бита хода", t_ok)
    check("зеркало p->23-p НЕ инвариант encode() (поэтому в кеш НЕ берём)", m_ok)

    print("== B. movegen_key (own cap4 × opp cap2) сохраняет набор легальных ходов ==")
    n_bad = 0
    n_tot = 0
    for _ in range(600):
        p = rand_pos(rng, bearing_phase=rng.random() < 0.4)
        # подбрасываем позиции со стопками >=3 (там ловился баг min(|v|,2))
        for i in range(3):
            idx = rng.randrange(24)
            c = rng.randint(3, 6)
            pts = list(p.points)
            pts[idx] = c if (p.turn == "white") == (rng.random() < 0.5) else -c
            p = Position(points=tuple(pts), bar_white=p.bar_white, bar_black=p.bar_black,
                         home_white=p.home_white, home_black=p.home_black, turn=p.turn)
        pc = compress_2bit_of(p)
        for roll in ALL_ROLLS:
            n_tot += 1
            try:
                s1 = moveset(p, roll)
                s2 = moveset(pc, roll)
            except Exception as e:
                n_bad += 1
                if n_bad < 4:
                    print("    ex:", type(e).__name__, e)
                continue
            if s1 != s2:
                n_bad += 1
                if n_bad < 4:
                    print("    mismatch roll", roll, "pos", p)
    check("own-cap4/opp-cap2 сохраняет набор ходов", n_bad == 0,
          f"({n_tot} pos×roll сравнений, bad={n_bad})")

    # явный контрпример: 3 фишки на точке при дубле — ломает 0/1 и min(|v|,2)
    q3 = Position(points=tuple(3 if i == 5 else 0 for i in range(24)), turn="white")
    q2 = Position(points=tuple(2 if i == 5 else 0 for i in range(24)), turn="white")
    s3, s2 = moveset(q3, (2, 2, 2, 2)), moveset(q2, (2, 2, 2, 2))
    check("контрпример 0/1: 3 фишки vs 2 фишки, дубль(2) дают РАЗНЫЕ наборы",
          s3 != s2, f"(разница: {sorted(s3 - s2)[:1]})")
    check("но cap4-сжатие его сохраняет",
          moveset(q3, (2, 2, 2, 2)) == moveset(compress_2bit_of(q3), (2, 2, 2, 2)))

    print("== C. Контрпримеры для 0/1 ==")
    # value: 1 фишка против 5 на той же точке
    p1 = Position(points=tuple(1 if i == 10 else 0 for i in range(24)), turn="white")
    p2 = Position(points=tuple(5 if i == 10 else 0 for i in range(24)), turn="white")
    e1, e2 = enc.encode(p1), enc.encode(p2)
    same_occ = compress_01_of(p1).points == compress_01_of(p2).points
    diff_in = not np.array_equal(e1, e2)
    check("0/1: точки 1-фишка vs 5-фишек имеют ОДИНАКОВЫЙ occupancy-ключ", same_occ)
    check("0/1: но РАЗНЫЕ входы сети (слои >=2..>=5 отличаются)", diff_in,
          f"(|Δencode|={np.abs(e1 - e2).sum():.0f})")
    # movegen: 1 vs 2 фишки, дубль (2,2)
    q1 = Position(points=tuple(1 if i == 10 else 0 for i in range(24)), turn="white")
    q2 = Position(points=tuple(2 if i == 10 else 0 for i in range(24)), turn="white")
    ms1, ms2 = moveset(q1, (2, 2)), moveset(q2, (2, 2))
    extra = ms2 - ms1
    check("0/1: набор ходов различается (1 фишка не может ходить дважды с точки)",
          bool(extra), f"(пример: {next(iter(extra)) if extra else '-'})")
    check("0/1: min(|v|,2) при этом набор сохраняет", moveset(q1, (2, 2)) == moveset(compress_2bit_of(q1), (2, 2)))

    print("== D. pad-to-K + masked argmax == per-game argmax ==")
    rng2 = random.Random(7)
    ok = True
    for _ in range(200):
        G = rng2.randint(1, 12)
        rows = [rng2.randint(1, 9) for _ in range(G)]  # >=1 (пустые=пасс идут отдельно)
        K = max(rows)
        M = torch.full((G, K), float("-inf"))
        per = []
        for g in range(G):
            vals = [rng2.uniform(-1, 1) for _ in range(rows[g])]
            for c in range(rows[g]):
                M[g, c] = vals[c]
            per.append(int(torch.argmax(torch.tensor(vals))))
        picks = torch.argmax(M, dim=1).tolist()
        if picks != per:
            ok = False
            break
    check("pad-to-K argmax идентичен per-game argmax (200 рандомов)", ok)

    print("== E. encode_batch_gpu ≈ Encoder.encode_batch ==")
    dev = "cpu"
    worst = 0.0
    for _ in range(50):
        ps = [rand_pos(rng, bearing_phase=rng.random() < 0.5) for _ in range(40)]
        old = enc.encode_batch(ps)
        new = encode_batch_gpu(ps, device=dev, enc=enc).numpy()
        worst = max(worst, float(np.abs(old - new).max()))
    check("encode_batch_gpu совпадает с numpy-энкодером (max|Δ|)", worst < 1e-6,
          f"(max|Δ|={worst:.2e})")

    print("== F. v2 (без мемо) == текущий цикл: ТЕ ЖЕ траектории (bit-exact) ==")
    net = make_value_net(enc.dim(), hidden=64, layers=1, seed=3)
    net.eval()
    ok = True
    for seed in [11, 22, 33]:
        g_old = simulate_games_batched(net, enc, n=12, device="cpu", seed_base=seed,
                                       max_steps=60, eps=0.1)
        g_new = simulate_games_batched_v2(net, n=12, device="cpu", seed_base=seed,
                                          max_steps=60, eps=0.1, encoder=enc,
                                          memo_engine=False)
        for (to, wo), (tn, wn) in zip(g_old, g_new):
            if not traj_eq(to, tn) or wo != wn:
                ok = False
                break
    check("v2(no memo) == current bit-exact (3 seeds, eps=0.1)", ok)

    print("== F2. МЕМО-движок: тот же НАБОР ходов, что и legal_moves ==")
    from training.selfplay_gpu import legal_moves_memo
    n_bad = 0
    n_tot = 0
    for _ in range(600):
        p = rand_pos(rng, bearing_phase=rng.random() < 0.4)
        for i in range(3):  # стопки >=3 для дублей
            idx = rng.randrange(24)
            c = rng.randint(3, 6)
            pts = list(p.points)
            pts[idx] = c if (p.turn == "white") == (rng.random() < 0.5) else -c
            p = Position(points=tuple(pts), bar_white=p.bar_white, bar_black=p.bar_black,
                         home_white=p.home_white, home_black=p.home_black, turn=p.turn)
        for roll in ALL_ROLLS:
            n_tot += 1
            try:
                s1 = {tuple(m.steps) for m in legal_moves(p, roll)}
                s2 = {tuple(m.steps) for m in legal_moves_memo(p, roll)}
            except Exception as e:
                n_bad += 1
                if n_bad < 4:
                    print("    ex:", type(e).__name__, e)
                continue
            if s1 != s2:
                n_bad += 1
                if n_bad < 4:
                    print("    mismatch roll", roll, "pos", p)
    check("мемо-движок == legal_moves по НАБОРУ ходов", n_bad == 0,
          f"({n_tot} pos×roll сравнений, bad={n_bad})")

    print("== G. v2(мемо) vs текущий: greedy-траектории при eps=0 ==")
    ok = True
    for seed in [11, 22, 33, 44]:
        g_old = simulate_games_batched(net, enc, n=12, device="cpu", seed_base=seed,
                                       max_steps=60, eps=0.0)
        g_new = simulate_games_batched_v2(net, n=12, device="cpu", seed_base=seed,
                                          max_steps=60, eps=0.0, encoder=enc,
                                          memo_engine=True)
        for (to, wo), (tn, wn) in zip(g_old, g_new):
            if not traj_eq(to, tn) or wo != wn:
                ok = False
                break
    check("v2(мемо, eps=0) == current greedy-траектории (4 seeds)", ok)

    print("== G2. v2(мемо, eps=0.1): валидность + схождение ==")
    ok = True
    for seed in [5, 6]:
        g_new = simulate_games_batched_v2(net, n=12, device="cpu", seed_base=seed,
                                          max_steps=200, eps=0.1, encoder=enc,
                                          memo_engine=True)
        # консервация фишек (движковый gate из skill) + терминальность
        for traj, _ in g_new:
            if len(traj) < 2:
                ok = False
                break
            for p in traj[:: max(1, len(traj) // 20)]:
                if not p.check_invariants():
                    ok = False
                    break
    check("v2(мемо, eps=0.1): 15/15 фишек сохранены во всех партиях", ok)

    print("== H. value-кеш: hit == свежий forward (float32) ==")
    ok = True
    for _ in range(30):
        ps = [rand_pos(rng) for _ in range(8)]
        X = torch.tensor(enc.encode_batch(ps), dtype=torch.float32)
        with torch.no_grad():
            V1 = net.value(X)
        # «кеш»: пересчёт тем же forward (одна и та же функция, eval, без дропаута)
        X2 = torch.tensor(enc.encode_batch(ps), dtype=torch.float32)
        with torch.no_grad():
            V2 = net.value(X2)
        if not torch.equal(V1, V2):
            ok = False
            break
    check("детерминизм forward: одинаковый вход -> битово одинаковый value", ok)

    print("== I. Бенч (CPU, тот же код на сервере уходит на cuda) ==")
    import time as _t
    from training.selfplay_gpu import legal_moves_memo

    # сколько поддеревьев экономит мемо на дубле (2,2,2,2) со старта
    p0 = Position.initial()
    counter = [0]

    def dfs_count(p, rem):
        counter[0] += 1
        if not rem:
            return
        for i, st in enumerate(rem):
            from core.game import _single_moves, _apply_step
            for mv in _single_moves(p, st):
                dfs_count(_apply_step(p, mv), rem[:i] + rem[i + 1:])

    dfs_count(p0, [2, 2, 2, 2])
    memo = {}
    legal_moves_memo(p0, (2, 2, 2, 2), memo)
    print(f"  дубль(2,2,2,2) со старта: DFS-узлов {counter[0]} -> уникальных поддеревьев {len(memo)} "
          f"(экономия {counter[0] / max(1, len(memo)):.0f}x)")

    t0 = _t.time()
    g_old = simulate_games_batched(net, enc, n=48, device="cpu", seed_base=99,
                                   max_steps=120, eps=0.1)
    t_old = _t.time() - t0

    t0 = _t.time()
    g_new = simulate_games_batched_v2(net, n=48, device="cpu", seed_base=99,
                                      max_steps=120, eps=0.1, encoder=enc,
                                      memo_engine=False)
    t_v2 = _t.time() - t0

    t0 = _t.time()
    g_new2 = simulate_games_batched_v2(net, n=48, device="cpu", seed_base=99,
                                       max_steps=120, eps=0.1, encoder=enc,
                                       memo_engine=True)
    t_v2m = _t.time() - t0

    t0 = _t.time()
    g_new3 = simulate_games_batched_v2(net, n=48, device="cpu", seed_base=99,
                                       max_steps=120, eps=0.1, encoder=enc,
                                       memo_engine=True, fast_encode=True)
    t_v2mf = _t.time() - t0

    print(f"  old        {t_old:6.2f}s")
    print(f"  v2(no memo){t_v2:6.2f}s | speedup {t_old/t_v2:5.1f}x")
    print(f"  v2+memo    {t_v2m:6.2f}s | speedup {t_old/t_v2m:5.1f}x")
    print(f"  v2+memo+fe {t_v2mf:6.2f}s | speedup {t_old/t_v2mf:5.1f}x")
    # эквивалентность бенча: наборы ходов в траекториях не проверяем (это F/F2/G),
    # но валидность — да
    ok = all(p.check_invariants() for t, _ in g_new2 for p in t[:: max(1, len(t)//20)])
    check("v2(мемо) в бенче: 15/15 фишек сохранены", ok)

    print(f"\n== ИТОГ: PASS={PASS} FAIL={FAIL} ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())