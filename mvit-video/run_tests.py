import torch
from torch import nn
# from torch.nn.init import trunc_normal_
from torch.nn.init import normal_, xavier_normal_, ones_, uniform_, orthogonal_, constant_
from torch import testing

from attention import MultiScaleAttention

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

    # +1 for the class emebedding
    x = torch.rand((1, T * H * W + 1, args["dim"])).to(dtype=torch.float32)
    y1 = attn1(x.clone(), thw_shape=thw_shape)
    y2 = attn2(x.clone(), thw_shape=thw_shape)

    testing.assert_close(y1, y2)

	# 2426
	# torch.Size([1, 25089, 96])
	# torch.float32
	# [8, 56, 56]

multi_scale_attn()