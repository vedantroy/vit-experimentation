import torch
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
    assert tensor.ndim == 4

    # TODO: remove cls_embed
    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, n_heads, L, dim_per_head = tensor.shape

    T, H, W = thw_shape
    tensor = rearrange(
        tensor, "b heads (T H W) d_head -> (b heads) d_head T H W", T=T, H=H, W=W
    )
    tensor = tensor.contiguous()
    dbg["q_pre_pool"] = tensor.clone()

    # TODO: Fully understand where query / kv pooling is utilized
    # This does not always reduce size
    tensor = pool(tensor)

    all_heads, dim_per_head, *thw_shape = tensor.shape
    thw_shape = list(thw_shape)
    assert len(thw_shape) == 3

    tensor = rearrange(
        tensor, "(b heads) d_head T H W -> b heads (T H W) d_head", b=B, heads=n_heads
    )

    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)

    if norm is not None:
        tensor = norm(tensor)

    return tensor, thw_shape


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
        assert len(kernel_kv) > 0, "kernel_q must be non-empty"
        assert mode == "conv", f"Only conv supported"
        assert not pool_first, f"Pooling 1st not supported"
        assert rel_pos_spatial and rel_pos_temporal, f"Only rel_pos supported"
        assert residual_pooling, "Must use residual_pooling"
        assert not separate_qkv, f"qkv must not be separate"

        # TODO: remove CLS embed
        self.has_cls_embed = has_cls_embed
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim_out * 3, bias=True)

        # From "Attention Is All You Need":
        # > We compute the dot products of the
        # > query with all keys, divide each by √dk,
        head_dim = dim_out // num_heads
        # 3D convs use channel, T, H, W
        # here dim_conv = channel
        dim_conv = head_dim
        self.scale = head_dim ** 0.5

        # Guessing this is the necessary padding
        # to preserve T/W/H
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.pool_q = nn.Conv3d(
            dim_conv,
            dim_conv,
            kernel_q,
            stride=stride_q,
            padding=padding_q,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = norm_layer(dim_conv)

        self.pool_k = nn.Conv3d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
        
        self.norm_k = norm_layer(dim_conv)
        self.pool_v = nn.Conv3d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
        )
        self.norm_v = norm_layer(dim_conv)

    def forward(self, x, thw_shape, dbg=None):
        dbg = default(dbg, {})
        B, THW, dim = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(
            qkv,
            "b thw (qkv heads dim_per_head) -> qkv b heads thw dim_per_head",
            qkv=3,
            heads=self.num_heads,
        )
        q, k, v = qkv

        _B, heads, _THW, dim_per_head = q.shape
        assert B == _B and dim == heads * dim_per_head and THW == _THW
        dbg["qkv"] = qkv

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q,
            dbg=dbg,
        )
        dbg["q"] = q

        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        THW = q.shape[2]
        attn = (q * self.scale) @ rearrange(
            k, "b heads thw dim_per_head -> b heads dim_per_head thw"
        )
        dbg["attn"] = attn.clone()