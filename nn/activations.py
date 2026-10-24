"""Activation function modules."""

from engine.tensor import Tensor
from nn.module import Module


class ReLU(Module):
    """Rectified Linear Unit: max(0, x)."""
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    """Hyperbolic tangent."""
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    """Logistic sigmoid: 1 / (1 + exp(-x))."""
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class GELU(Module):
    """Gaussian Error Linear Unit."""
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()
