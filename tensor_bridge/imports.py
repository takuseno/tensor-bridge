from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import jax
    import nnabla
    import torch

__all__ = ["get_torch", "get_jax", "get_nnabla"]


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


def get_nnabla() -> Optional["nnabla"]:
    try:
        import nnabla

        return nnabla
    except ImportError:
        return None
