"""Base Module class for building neural network layers."""

from __future__ import annotations
from typing import List
from engine.tensor import Tensor


class Module:
    """Base class for all neural-network modules.

    Subclasses must implement ``forward()``.  Parameters are discovered via
    ``parameters()`` which recursively walks ``__dict__`` looking for
    :class:`Tensor` objects with ``requires_grad=True`` and child
    :class:`Module` instances.
    """

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters (recursively)."""
        params: List[Tensor] = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        """Set all parameter gradients to None."""
        for p in self.parameters():
            p.zero_grad()


class Sequential(Module):
    """A sequential container that chains modules."""

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
