from turtle import forward
from torch import nn

from einops import rearrange

# Simplified helper functions from lucidrain
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        separate_qkv=False,
    ):
        super().__init__()

        # Most/all? of the models in PySlowFast have bias w/ QKV
        # even though original attention paper doesn't
        assert qkv_bias, f"qkv must have bias"
        assert mode == "conv", f"Only conv supported"
        assert not pool_first, f"Pooling 1st not supported"
        assert rel_pos_spatial and rel_pos_temporal, f"Only rel_pos supported"
        assert residual_pooling, "Must use residual_pooling"
        assert not separate_qkv, f"qkv must not be separate"

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim_out * 3, bias=True)

    def forward(self, x, thw_shape, dbg=None):
        dbg = default(dbg, {})

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b thw (qkv heads rest) -> b thw qkv heads rest", qkv=3, heads=self.num_heads)
        qkv = rearrange(qkv, "b thw qkv heads rest -> qkv b heads thw rest")
        q, k, v = qkv

        dbg["qkv"] = qkv