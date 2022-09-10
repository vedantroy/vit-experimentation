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

def attention_pool(tensor, pool, thw_shape, has_cls_embed, norm, dbg=None):
    dbg = default(dbg, {})

    # TODO: remove cls_embed
    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, n_heads, L, dim_per_head = tensor.shape

    T, H, W = thw_shape
    tensor = rearrange(tensor, "b heads (T H W) d_head -> (b heads) T H W d_head", T=T, H=H, W=W)
    tensor = rearrange(tensor, "bheads T H W d_head -> bheads d_head T H W")
    tensor = tensor.contiguous()
    dbg["q_pre_pool"] = tensor.clone()

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
        assert len(kernel_q) > 0, "kernel_q must be non-empty"
        assert mode == "conv", f"Only conv supported"
        assert not pool_first, f"Pooling 1st not supported"
        assert rel_pos_spatial and rel_pos_temporal, f"Only rel_pos supported"
        assert residual_pooling, "Must use residual_pooling"
        assert not separate_qkv, f"qkv must not be separate"

        # TODO: remove CLS embed
        self.has_cls_embed = has_cls_embed
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim_out * 3, bias=True)


        # Guessing this is the necessary padding
        # to preserve T/W/H
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        dim_conv = dim_out // num_heads
        self.pool_q = nn.Conv3d(
                dim_conv,
                dim_conv,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim_conv,
                bias=False
        )
        self.norm_q = norm_layer(dim_conv)

    def forward(self, x, thw_shape, dbg=None):
        dbg = default(dbg, {})
        B, THW, dim = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b thw (qkv heads dim_per_head) -> qkv b heads thw dim_per_head", qkv=3, heads=self.num_heads)
        q, k, v = qkv

        _B, heads, _THW, dim_per_head = q.shape
        assert B == _B and dim == heads * dim_per_head and THW == _THW
        dbg["qkv"] = qkv

        attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q,
            dbg=dbg,
        )