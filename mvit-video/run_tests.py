import torch
from torch import nn

# from torch.nn.init import trunc_normal_
from torch.nn.init import (
    normal_,
    xavier_normal_,
    ones_,
    uniform_,
    orthogonal_,
    constant_,
)
from torch import testing

from attention import MultiScaleAttention
from my_attention import MultiScaleAttention as MyMultiScaleAttention


def assert_shape(x, shape):
    actual = tuple(x.shape)
    if actual != shape:
        raise ValueError(f"{actual} != {shape}")


torch.use_deterministic_algorithms(mode=True)


def deterministic_fill_(p):
    gen = torch.Generator()
    gen.manual_seed(42)

    shape = p.shape
    weights = torch.rand(shape, generator=gen)
    p.data = weights


def deterministic_init(m):
    params = 0
    for n, p in m.named_parameters():
        deterministic_fill_(p)
        params += 1
    print(f"{params} params deterministically initialized")


def model_args():
    args = dict(
        dim=96,
        dim_out=96,
        input_size=[8, 56, 56],
        num_heads=1,
        qkv_bias=True,
        drop_rate=0.0,
        kernel_q=[3, 3, 3],
        kernel_kv=[3, 3, 3],
        stride_q=[1, 1, 1],
        stride_kv=[1, 8, 8],
        has_cls_embed=True,
        mode="conv",
        pool_first=False,
        rel_pos_spatial=True,
        rel_pos_temporal=True,
        rel_pos_zero_init=False,
        residual_pooling=True,
        separate_qkv=False,
    )
    return args


def multi_scale_attn():
    args = model_args()

    attn1 = MultiScaleAttention(**args)
    attn2 = MultiScaleAttention(**args)
    deterministic_init(attn1)
    deterministic_init(attn2)

    T, H, W = 8, 56, 56
    thw_shape = [T, H, W]

    # Ensure the deterministic init works
    # +1 for the class embedding
    x = torch.rand((1, T * H * W + 1, args["dim"])).to(dtype=torch.float32)
    with torch.no_grad():
        y1 = attn1(x.clone(), thw_shape=thw_shape)
        y2 = attn2(x.clone(), thw_shape=thw_shape)
    testing.assert_close(y1, y2)
    print("deterministic init passed")

    my_attn = MyMultiScaleAttention(**args)
    deterministic_init(my_attn)

    # Ensure qkv works
    with torch.no_grad():
        print("Starting tests ...")
        actual_dbg = {}
        attn1(x.clone(), thw_shape=thw_shape, dbg=actual_dbg)
        my_dbg = {}
        my_attn(x.clone(), thw_shape=thw_shape, dbg=my_dbg)

        actual_qkv = actual_dbg["qkv"]
        my_qkv = my_dbg["qkv"]
        assert_shape(actual_qkv, my_qkv.shape)
        print("qkv shape passed")
        testing.assert_close(my_qkv, actual_qkv)
        print("qkv match passed")

        actual_q_pre_pool = actual_dbg["q_pre_pool"]
        my_q_pre_pool = my_dbg["q_pre_pool"]
        testing.assert_close(actual_q_pre_pool, my_q_pre_pool)
        print("q_pre_pool passed")

        actual_q = actual_dbg["q"]
        my_q = my_dbg["q"]
        testing.assert_close(actual_q, my_q)
        print("pooled_q passed")

        actual_attn_matrix = actual_dbg["attn"]
        my_attn_matrix = my_dbg["attn"]
        testing.assert_close(actual_attn_matrix, my_attn_matrix)
        print("pooled_q passed")


multi_scale_attn()
