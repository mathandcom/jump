"""JUMP attack package."""

from .core.attack.runner import main as eval_main
from .core.training.runner import main as train_main

__all__ = ["eval_main", "train_main"]
