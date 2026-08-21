"""ML-модели и обучение для агента нард.

- model.net — value/policy сети (TD-Gammon-стиль MLP).
- training.selfplay — генерация партий greedy-политикой.
- training.td — TD(λ)-обучение (BP-baseline; predictive-coding/EP — следующий этап).
"""

from . import net  # noqa: F401