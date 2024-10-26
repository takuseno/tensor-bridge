from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    import jax
    import torch

__all__ = ["get_torch", "get_jax"]


def get_torch() -> Optional["torch"]:
    try:
        import torch

        return torch
    except ImportError:
        return None


def get_jax() -> Optional["jax"]:
    try:
        import jax

        return jax
    except ImportError:
        return None
