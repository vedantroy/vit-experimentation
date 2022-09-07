from types import SimpleNamespace

import torch

from mvit_model import MViT

if __name__ == "__main__":
    cfg_mvit = dict(
        PATCH_2D=True,
        ZERO_DECAY_POS_CLS=False,
        MODE="conv",
        CLS_EMBED_ON=True,
        PATCH_KERNEL=[7, 7],
        PATCH_STRIDE=[4, 4],
        PATCH_PADDING=[3, 3],
        EMBED_DIM=96,
        NUM_HEADS=1,
        MLP_RATIO=4.0,
        QKV_BIAS=True,
        DROPPATH_RATE=0.1,
        DEPTH=16,
        NORM="layernorm",
        DIM_MUL=[[1, 2.0], [3, 2.0], [14, 2.0]],
        HEAD_MUL=[[1, 2.0], [3, 2.0], [14, 2.0]],
        POOL_KVQ_KERNEL=[1, 3, 3],
        POOL_KV_STRIDE_ADAPTIVE=[1, 4, 4],
        POOL_Q_STRIDE=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
        # Not in MVIT_B_16_CONV.yaml,
        # but found in MVITv2_B.yaml
        USE_ABS_POS=False,
        DIM_MUL_IN_ATT=True,
        # Set to False to avoid an assertion error
        REL_POS_SPATIAL=False,
        # Found in MVITv2_L_40x3.yaml
        POOL_FIRST=False,
        RESIDUAL_POOLING=True,
        # Not found anywhere, but seems to be generally False
        REL_POS_ZERO_INIT=False
    )

    cfg_model = dict(
        NUM_CLASSES=1000,
        ARCH="mvit",
        MODEL_NAME=MViT,
        LOSS_FUNC="soft_cross_entropy",
        DROPOUT_RATE=0.0,
        # Not in yaml
        ACT_CHECKPOINT=False,
        # default is softmax
        HEAD_ACT="softmax"
    )

    cfg_data = dict(
        MEAN=[0.485, 0.456, 0.406],
        STD=[0.229, 0.224, 0.225],
        NUM_FRAMES=1,
        TRAIN_CROP_SIZE=224,
        TEST_CROP_SIZE=224,
        INPUT_CHANNEL_NUM=[3],
    )

    cfg_mvit = SimpleNamespace(**cfg_mvit)
    cfg_model = SimpleNamespace(**cfg_model)
    cfg_data = SimpleNamespace(**cfg_data)

    cfg = dict(
	    MVIT=cfg_mvit,
	    MODEL=cfg_model,
        DATA=cfg_data
    )

    cfg = SimpleNamespace(**cfg)

    model = MViT(cfg)
    shape = (2, 3, 1, cfg_data.TRAIN_CROP_SIZE, cfg_data.TRAIN_CROP_SIZE)
    x = torch.randn(shape)
    y = model(x)