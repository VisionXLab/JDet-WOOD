# Ported from MobileSAM `modeling/common.py`.
# for Point2RBox-v3-jittor. Inference-only.
#
# UNVERIFIED-API (check once env is ready):
#   - nn.GELU exists as a class in this Jittor build (else use a lambda over jt.nn.gelu)

import jittor as jt
from jittor import nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        mlp_dim,
        act=nn.GELU,
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def execute(self, x):
        return self.lin2(self.act(self.lin1(x)))


# From detectron2 / ConvNeXt (see upstream header).
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        # torch: nn.Parameter; in Jittor a jt.Var attribute is a parameter.
        self.weight = jt.ones(num_channels)
        self.bias = jt.zeros(num_channels)
        self.eps = eps

    def execute(self, x):
        u = x.mean(1, keepdims=True)
        s = ((x - u) ** 2).mean(1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
