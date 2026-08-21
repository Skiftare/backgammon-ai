"""Тест ML-стека: энкодер → value-сеть → self-play → TD-обучение.

Интеграционный смоук: одна траектория greedy-политики и один шаг TD(λ)-обучения
(BP-baseline). Проверяем, что стек сходится без NaN и loss — положительное число.
"""

import random
import sys

import numpy as np
import pytest
import torch

from core.board import Position
from core.features import Encoder
from model.net import make_value_net
from training.selfplay import play_one_game
from training.td import train_episode


def test_td_loss_finite_and_positive():
    torch.manual_seed(0)
    enc = Encoder()
    net = make_value_net(enc.dim(), hidden=64)
    net.eval()

    traj, winner = play_one_game(net, enc, rng=random.Random(7), max_steps=80)
    assert len(traj) >= 2
    assert winner in ("white", "black")

    loss = train_episode(net, enc, traj, winner, lr=1e-3)
    assert torch.isfinite(torch.tensor(loss))
    assert loss > 1e-9


def test_backward_no_nan():
    enc = Encoder()
    net = make_value_net(enc.dim(), hidden=64)
    x = torch.tensor(enc.encode(Position.initial()), dtype=torch.float32).unsqueeze(0)
    v = net.value(x)
    loss = (v - 1.0) ** 2
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())