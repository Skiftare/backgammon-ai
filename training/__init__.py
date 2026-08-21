"""Обучение агента нард.

- selfplay.py: генерация партии (greedy по value-сети).
- td.py: TD(λ) обучение value-сети (BP-baseline).
"""

from . import selfplay, td  # noqa: F401