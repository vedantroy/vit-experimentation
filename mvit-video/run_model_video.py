from types import SimpleNamespace
from math import inf

import torch

from video_model_builder2 import MViT

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

cfg = dict(
  AUG=dict(
    AA_TYPE="rand-m7-n4-mstd0.5-inc1",
    COLOR_JITTER=0.4,
    ENABLE=True,
    GEN_MASK_LOADER=False,
    INTERPOLATION="bicubic",
    MASK_FRAMES=False,
    MASK_RATIO=0.0,
    MASK_TUBE=False,
    MASK_WINDOW_SIZE=[8, 7, 7],
    MAX_MASK_PATCHES_PER_BLOCK=None,
    NUM_SAMPLE=2,
    RE_COUNT=1,
    RE_MODE="pixel",
    RE_PROB=0.25,
    RE_SPLIT=False,
  ),
  BN=dict(
    GLOBAL_SYNC=False,
    NORM_TYPE="batchnorm",
    NUM_BATCHES_PRECISE=200,
    NUM_SPLITS=1,
    NUM_SYNC_DEVICES=1,
    USE_PRECISE_STATS=False,
    WEIGHT_DECAY=0.0,
  ),
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
  DATA=dict(
    COLOR_RND_GRAYSCALE=0.0,
    DECODING_BACKEND="torchvision",
    DECODING_SHORT_SIZE=256,
    DUMMY_LOAD=False,
    ENSEMBLE_METHOD="sum",
    IN22K_TRAINVAL=False,
    IN22k_VAL_IN1K="",
    INPUT_CHANNEL_NUM=[3],
    INV_UNIFORM_SAMPLE=False,
    IN_VAL_CROP_RATIO=0.875,
    LOADER_CHUNK_OVERALL_SIZE=0,
    LOADER_CHUNK_SIZE=0,
    MEAN=[0.45, 0.45, 0.45],
    MULTI_LABEL=False,
    NUM_FRAMES=32,
    PATH_LABEL_SEPARATOR="",
    PATH_PREFIX="",
    PATH_TO_DATA_DIR="",
    PATH_TO_PRELOAD_IMDB="",
    RANDOM_FLIP=True,
    REVERSE_INPUT_CHANNEL=False,
    SAMPLING_RATE=3,
    SKIP_ROWS=0,
    SSL_BLUR_SIGMA_MAX=[0.0, 2.0],
    SSL_BLUR_SIGMA_MIN=[0.0, 0.1],
    SSL_COLOR_BRI_CON_SAT=[0.4, 0.4, 0.4],
    SSL_COLOR_HUE=0.1,
    SSL_COLOR_JITTER=False,
    SSL_MOCOV2_AUG=False,
    STD=[0.225, 0.225, 0.225],
    TARGET_FPS=30,
    TEST_CROP_SIZE=224,
    TIME_DIFF_PROB=0.0,
    TRAIN_CROP_NUM_SPATIAL=1,
    TRAIN_CROP_NUM_TEMPORAL=1,
    TRAIN_CROP_SIZE=224,
    TRAIN_JITTER_ASPECT_RELATIVE=[0.75, 1.3333],
    TRAIN_JITTER_FPS=0.0,
    TRAIN_JITTER_MOTION_SHIFT=False,
    TRAIN_JITTER_SCALES=[256, 320],
    TRAIN_JITTER_SCALES_RELATIVE=[0.08, 1.0],
    TRAIN_PCA_EIGVAL=[0.225, 0.224, 0.229],
    TRAIN_PCA_EIGVEC=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.814], [-0.5836, -0.6948, 0.4203]],
    USE_OFFSET_SAMPLING=True,
  ),
  DATA_LOADER=dict(
    ENABLE_MULTI_THREAD_DECODE=False,
    NUM_WORKERS=8,
    PIN_MEMORY=True,
  ),
  DEMO=dict(
    BUFFER_SIZE=0,
    CLIP_VIS_SIZE=10,
    COMMON_CLASS_NAMES=['watch (a person)', 'talk to (e.g., self, a person, a group)', 'listen to (a person)', 'touch (an object)', 'carry/hold (an object)', 'walk', 'sit', 'lie/sleep', 'bend/bow (at the waist)'],
    COMMON_CLASS_THRES=0.7,
    DETECTRON2_CFG="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    DETECTRON2_THRESH=0.9,
    DETECTRON2_WEIGHTS="detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
    DISPLAY_HEIGHT=0,
    DISPLAY_WIDTH=0,
    ENABLE=False,
    FPS=30,
    GT_BOXES="",
    INPUT_FORMAT="BGR",
    INPUT_VIDEO="",
    LABEL_FILE_PATH="",
    NUM_CLIPS_SKIP=0,
    NUM_VIS_INSTANCES=2,
    OUTPUT_FILE="",
    OUTPUT_FPS=-1,
    PREDS_BOXES="",
    SLOWMO=1,
    STARTING_SECOND=900,
    THREAD_ENABLE=False,
    UNCOMMON_CLASS_THRES=0.3,
    VIS_MODE="thres",
    WEBCAM=-1,
  ),
  DETECTION=dict(
    ALIGNED=True,
    ENABLE=False,
    ROI_XFORM_RESOLUTION=7,
    SPATIAL_SCALE_FACTOR=16,
  ),
  DIST_BACKEND= "nccl",
  LOG_MODEL_INFO= True,
  LOG_PERIOD=10,
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
    DROPOUT_RATE=0.5,
    FC_INIT_STD=0.01,
    FP16_ALLREDUCE=False,
    FROZEN_BN=False,
    HEAD_ACT="softmax",
    LOSS_FUNC="soft_cross_entropy",
    MODEL_NAME="MViT",
    MULTI_PATHWAY_ARCH=['slowfast'],
    NUM_CLASSES=400,
    SINGLE_PATHWAY_ARCH=['2d', 'c2d', 'i3d', 'slow', 'x3d', 'mvit', 'maskmvit'],
  ),
  MULTIGRID=dict(
    BN_BASE_SIZE=8,
    DEFAULT_B=0,
    DEFAULT_S=0,
    DEFAULT_T=0,
    EPOCH_FACTOR=1.5,
    EVAL_FREQ=3,
    LONG_CYCLE=False,
    LONG_CYCLE_FACTORS=[(0.25, 0.7071067811865476), (0.5, 0.7071067811865476), (0.5, 1), (1, 1)],
    LONG_CYCLE_SAMPLING_RATE=0,
    SHORT_CYCLE=False,
    SHORT_CYCLE_FACTORS=[0.5, 0.7071067811865476],
  ),
  MVIT=dict(
    # CLS_EMBED_ON=True,
    CLS_EMBED_ON=False,
    DEPTH=24,
    DIM_MUL=[[2, 2.0], [5, 2.0], [21, 2.0]],
    DIM_MUL_IN_ATT=False,
    DROPOUT_RATE=0.0,
    DROPPATH_RATE=0.3,
    EMBED_DIM=96,
    HEAD_INIT_SCALE=1.0,
    HEAD_MUL=[[2, 2.0], [5, 2.0], [21, 2.0]],
    LAYER_SCALE_INIT_VALUE=0.0,
    MLP_RATIO=4.0,
    MODE="conv",
    NORM="layernorm",
    NORM_STEM=False,
    NUM_HEADS=1,
    PATCH_2D=False,
    PATCH_KERNEL=[3, 7, 7],
    PATCH_PADDING=[1, 3, 3],
    PATCH_STRIDE=[2, 4, 4],
    POOL_FIRST=False,
    POOL_KVQ_KERNEL=None,
    POOL_KV_STRIDE=[],
    POOL_KV_STRIDE_ADAPTIVE=[1, 8, 8],
    POOL_Q_STRIDE=[[2, 1, 2, 2], [5, 1, 2, 2], [21, 1, 2, 2]],
    QKV_BIAS=True,
    REL_POS_SPATIAL=False,
    REL_POS_TEMPORAL=False,
    REL_POS_ZERO_INIT=False,
    RESIDUAL_POOLING=False,
    REV=dict(
      ENABLE=False,
      # doesn't matter since enable=false
      RESPATH_FUSE="concat",
    ),
    SEPARATE_QKV=False,
    SEP_POS_EMBED=True,
    USE_ABS_POS=True,
    USE_FIXED_SINCOS_POS=False,
    USE_MEAN_POOLING=False,
    ZERO_DECAY_POS_CLS=False,
  ),
  NONLOCAL=dict(
    GROUP=[[1], [1], [1], [1]],
    INSTANTIATION="dot_product",
    LOCATION=[[[]], [[]], [[]], [[]]],
    POOL=[[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]],
  ),
  NUM_GPUS=2,
  NUM_SHARDS=1,
  OUTPUT_DIR=".",
  RESNET=dict(
    DEPTH=50,
    INPLACE_RELU=True,
    NUM_BLOCK_TEMP_KERNEL=[[3], [4], [6], [3]],
    NUM_GROUPS=1,
    SPATIAL_DILATIONS=[[1], [1], [1], [1]],
    SPATIAL_STRIDES=[[1], [2], [2], [2]],
    STRIDE_1X1=False,
    TRANS_FUNC="bottleneck_transform",
    WIDTH_PER_GROUP=64,
    ZERO_INIT_FINAL_BN=False,
    ZERO_INIT_FINAL_CONV=False,
  ),
  SLOWFAST=dict(
    ALPHA=8,
    BETA_INV=8,
    FUSION_CONV_CHANNEL_RATIO=2,
    FUSION_KERNEL_SZ=5,
  ),
  SOLVER=dict(
    BASE_LR=0.0001,
    BASE_LR_SCALE_NUM_SHARDS=True,
    BETAS="(0.9, 0.999)",
    CLIP_GRAD_L2NORM=1.0,
    CLIP_GRAD_VAL=None,
    COSINE_AFTER_WARMUP=True,
    COSINE_END_LR=1e-06,
    DAMPENING=0.0,
    GAMMA=0.1,
    LARS_ON=False,
    LAYER_DECAY=1.0,
    LRS=[],
    LR_POLICY="cosine",
    MAX_EPOCH=200,
    MOMENTUM=0.9,
    NESTEROV=True,
    OPTIMIZING_METHOD="adamw",
    STEPS=[],
    STEP_SIZE=1,
    WARMUP_EPOCHS=30.0,
    WARMUP_FACTOR=0.1,
    WARMUP_START_LR=1e-06,
    WEIGHT_DECAY=0.05,
    ZERO_WD_1D_PARAM=True,
  ),
  TEST=dict(
    BATCH_SIZE=64,
    CHECKPOINT_FILE_PATH="",
    CHECKPOINT_TYPE="pytorch",
    DATASET="kinetics",
    ENABLE=True,
    NUM_ENSEMBLE_VIEWS=10,
    NUM_SPATIAL_CROPS=1,
    NUM_TEMPORAL_CLIPS=[],
    SAVE_RESULTS_PATH="",
  ),
  TRAIN=dict(
    AUTO_RESUME=True,
    BATCH_SIZE=16,
    CHECKPOINT_CLEAR_NAME_PATTERN="()",
    CHECKPOINT_EPOCH_RESET=False,
    CHECKPOINT_FILE_PATH="",
    CHECKPOINT_INFLATE=False,
    CHECKPOINT_IN_INIT=False,
    CHECKPOINT_PERIOD=10,
    CHECKPOINT_TYPE="pytorch",
    DATASET="kinetics",
    ENABLE=True,
    EVAL_PERIOD=10,
    KILL_LOSS_EXPLOSION_FACTOR=0.0,
    MIXED_PRECISION=False,
  ),
  VIS_MASK=dict(
    ENABLE=False,
  ),
  X3D=dict(
    BN_LIN5=False,
    BOTTLENECK_FACTOR=1.0,
    CHANNELWISE_3x3x3=True,
    DEPTH_FACTOR=1.0,
    DIM_C1=12,
    DIM_C5=2048,
    SCALE_RES2=False,
    WIDTH_FACTOR=1.0,
  )
)

if __name__ == "__main__":
   cfg = RecursiveNamespace(**cfg)
   model = MViT(cfg)

   fake_batch_size = 1
   input_data = torch.randn((1, fake_batch_size, 3, cfg.DATA.NUM_FRAMES, cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE))

   with torch.no_grad():
       model = model.cuda()
       input_data = input_data.cuda()
       y = model(input_data)
       print(y.shape)