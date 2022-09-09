from types import SimpleNamespace

import torch

from video_model_builder import MViT

# Taken from:
# https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
class RecursiveNamespace(SimpleNamespace):
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def config():
    from math import inf

    # Taken from running the model with a .yaml
    # (forget which one) & dumping the fully instantiated
    # config
    cfg = dict(
        MASK=dict(
            DECODER_DEPTH=0,
            DECODER_EMBED_DIM=512,
            DECODER_SEP_POS_EMBED=False,
            DEC_KV_KERNEL=[],
            DEC_KV_STRIDE=[],
            ENABLE=False,
            HEAD_TYPE="separate",
            MAE_ON=False,
            MAE_RND_MASK=False,
            NORM_PRED_PIXEL=True,
            PER_FRAME_MASKING=False,
            PRED_HOG=False,
            PRETRAIN_DEPTH=[15],
            SCALE_INIT_BY_DEPTH=False,
            TIME_STRIDE_LOSS=True,
        ),
        MIXUP=dict(
            ALPHA=0.8,
            CUTMIX_ALPHA=1.0,
            ENABLE=True,
            LABEL_SMOOTH_VALUE=0.1,
            PROB=1.0,
            SWITCH_PROB=0.5,
        ),
        MODEL=dict(
            ACT_CHECKPOINT=False,
            ARCH="mvit",
            DETACH_FINAL_FC=False,
            DROPCONNECT_RATE=0.0,
            DROPOUT_RATE=0.0,
            FC_INIT_STD=0.01,
            FP16_ALLREDUCE=False,
            FROZEN_BN=False,
            HEAD_ACT="softmax",
            LOSS_FUNC="soft_cross_entropy",
            MODEL_NAME="MViT",
            MULTI_PATHWAY_ARCH=["slowfast"],
            NUM_CLASSES=400,
            SINGLE_PATHWAY_ARCH=["2d", "c2d", "i3d", "slow", "x3d", "mvit", "maskmvit"],
        ),
        MULTIGRID=dict(
            BN_BASE_SIZE=8,
            DEFAULT_B=0,
            DEFAULT_S=0,
            DEFAULT_T=0,
            EPOCH_FACTOR=1.5,
            EVAL_FREQ=3,
            LONG_CYCLE=False,
            LONG_CYCLE_FACTORS=[
                (0.25, 0.7071067811865476),
                (0.5, 0.7071067811865476),
                (0.5, 1),
                (1, 1),
            ],
            LONG_CYCLE_SAMPLING_RATE=0,
            SHORT_CYCLE=False,
            SHORT_CYCLE_FACTORS=[0.5, 0.7071067811865476],
        ),
        MVIT=dict(
            CLS_EMBED_ON=True,
            DEPTH=16,
            DIM_MUL=[[1, 2.0], [3, 2.0], [14, 2.0]],
            DIM_MUL_IN_ATT=False,
            DROPOUT_RATE=0.0,
            DROPPATH_RATE=0.1,
            EMBED_DIM=96,
            HEAD_INIT_SCALE=0.001,
            HEAD_MUL=[[1, 2.0], [3, 2.0], [14, 2.0]],
            LAYER_SCALE_INIT_VALUE=0.0,
            MLP_RATIO=4.0,
            MODE="conv",
            NORM="layernorm",
            NORM_STEM=False,
            NUM_HEADS=1,

            # These are the parameters for the convolution that runs
            # on the input to turn it into an embedding
            PATCH_2D=False,
            PATCH_KERNEL=[3, 7, 7],
            PATCH_PADDING=[1, 3, 3],
            PATCH_STRIDE=[2, 4, 4],

            POOL_FIRST=False,
            POOL_KVQ_KERNEL=[3, 3, 3],
            POOL_KV_STRIDE=[],
            POOL_KV_STRIDE_ADAPTIVE=[1, 8, 8],
            POOL_Q_STRIDE=[
                [0, 1, 1, 1],
                [1, 1, 2, 2],
                [2, 1, 1, 1],
                [3, 1, 2, 2],
                [4, 1, 1, 1],
                [5, 1, 1, 1],
                [6, 1, 1, 1],
                [7, 1, 1, 1],
                [8, 1, 1, 1],
                [9, 1, 1, 1],
                [10, 1, 1, 1],
                [11, 1, 1, 1],
                [12, 1, 1, 1],
                [13, 1, 1, 1],
                [14, 1, 2, 2],
                [15, 1, 1, 1],
            ],
            QKV_BIAS=True,
            REL_POS_SPATIAL=True,
            REL_POS_TEMPORAL=True,
            REL_POS_ZERO_INIT=False,
            RESIDUAL_POOLING=True,
            REV=dict(
                BUFFER_LAYERS=[],
                ENABLE=False,
                PRE_Q_FUSION="avg",
                RESPATH_FUSE="concat",
                RES_PATH="conv",
            ),
            SEPARATE_QKV=False,
            SEP_POS_EMBED=True,
            USE_ABS_POS=False,
            USE_FIXED_SINCOS_POS=False,
            USE_MEAN_POOLING=True,
            ZERO_DECAY_POS_CLS=False,
        ),
        DATA={
            "COLOR_RND_GRAYSCALE": 0.0,
            "DECODING_BACKEND": "torchvision",
            "DECODING_SHORT_SIZE": 320,
            "DUMMY_LOAD": False,
            "ENSEMBLE_METHOD": "sum",
            "IN22K_TRAINVAL": False,
            "IN22k_VAL_IN1K": "",
            "INPUT_CHANNEL_NUM": [3],
            "INV_UNIFORM_SAMPLE": False,
            "IN_VAL_CROP_RATIO": 0.875,
            "LOADER_CHUNK_OVERALL_SIZE": 0,
            "LOADER_CHUNK_SIZE": 0,
            "MEAN": [0.45, 0.45, 0.45],
            "MULTI_LABEL": False,
            "NUM_FRAMES": 16,
            "PATH_LABEL_SEPARATOR": " ",
            "PATH_PREFIX": "",
            "PATH_TO_DATA_DIR": "",
            "PATH_TO_PRELOAD_IMDB": "",
            "RANDOM_FLIP": True,
            "REVERSE_INPUT_CHANNEL": False,
            "SAMPLING_RATE": 4,
            "SKIP_ROWS": 0,
            "SSL_BLUR_SIGMA_MAX": [0.0, 2.0],
            "SSL_BLUR_SIGMA_MIN": [0.0, 0.1],
            "SSL_COLOR_BRI_CON_SAT": [0.4, 0.4, 0.4],
            "SSL_COLOR_HUE": 0.1,
            "SSL_COLOR_JITTER": False,
            "SSL_MOCOV2_AUG": False,
            "STD": [0.225, 0.225, 0.225],
            "TARGET_FPS": 30,
            "TEST_CROP_SIZE": 224,
            "TIME_DIFF_PROB": 0.0,
            "TRAIN_CROP_NUM_SPATIAL": 1,
            "TRAIN_CROP_NUM_TEMPORAL": 1,
            "TRAIN_CROP_SIZE": 224,
            "TRAIN_JITTER_ASPECT_RELATIVE": [0.75, 1.3333],
            "TRAIN_JITTER_FPS": 0.0,
            "TRAIN_JITTER_MOTION_SHIFT": False,
            "TRAIN_JITTER_SCALES": [256, 320],
            "TRAIN_JITTER_SCALES_RELATIVE": [0.08, 1.0],
            "TRAIN_PCA_EIGVAL": [0.225, 0.224, 0.229],
            "TRAIN_PCA_EIGVEC": [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.814],
                [-0.5836, -0.6948, 0.4203],
            ],
            "USE_OFFSET_SAMPLING": True,
        },
        DETECTION={
            "ALIGNED": True,
            "ENABLE": False,
            "ROI_XFORM_RESOLUTION": 7,
            "SPATIAL_SCALE_FACTOR": 16,
        },
        CONTRASTIVE=dict(
            BN_MLP=False,
            BN_SYNC_MLP=False,
            DELTA_CLIPS_MAX=inf,
            DELTA_CLIPS_MIN=-inf,
            DIM=128,
            INTERP_MEMORY=False,
            KNN_ON=True,
            LENGTH=239975,
            LOCAL_SHUFFLE_BN=True,
            MEM_TYPE="1d",
            MLP_DIM=2048,
            MOCO_MULTI_VIEW_QUEUE=False,
            MOMENTUM=0.5,
            MOMENTUM_ANNEALING=False,
            NUM_CLASSES_DOWNSTREAM=400,
            NUM_MLP_LAYERS=1,
            PREDICTOR_DEPTHS=[],
            QUEUE_LEN=65536,
            SEQUENTIAL=False,
            SIMCLR_DIST_ON=True,
            SWAV_QEUE_LEN=0,
            T=0.07,
            TYPE="mem",
        ),
    )

    return RecursiveNamespace(**cfg)


# NONLOCAL:
#  GROUP: [[1], [1], [1], [1]]
#  INSTANTIATION: dot_product
#  LOCATION: [[[]], [[]], [[]], [[]]]
#  POOL: [[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]]


def main():
    cfg = config()
    model = MViT(cfg)
    # unclear why the leading 1, make batch size = 1
    shape = (
        1,
        1,
        3,
        cfg.DATA.NUM_FRAMES,
        cfg.DATA.TRAIN_CROP_SIZE,
        cfg.DATA.TRAIN_CROP_SIZE,
    )
    x = torch.randn(shape)
    y = model(x)
    print("Model finished running !")
    print(y.shape)


if __name__ == "__main__":
    main()
