GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1
export LD_PRELOAD="$(dirname $0)/stub_itt.so${LD_PRELOAD:+:$LD_PRELOAD}"

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S_Lite_mlp --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --no-syncbn \
    --mst \
    --compile_mode="default" \
    --amp \
    --val_amp \
    --pad_SUNRGBD \
    --no-use_seed \
    --continue_fpath checkpoints/NYUDepthv2_DFormerv2_S_Lite_mlp_20260531-224225/epoch-230_miou_51.7.pth

# config for DFormers on NYUDepthv2
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormer_v2_S
# local_configs.NYUDepthv2.DFormer_v2_B
# local_configs.NYUDepthv2.DFormer_v2_L
# local_configs.NYUDepthv2.DFormerv2_S_Lite_ham
# local_configs.NYUDepthv2.DFormerv2_S_Lite_mlp

# config for DFormers on SUNRGBD
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormer_v2_S
# local_configs.SUNRGBD.DFormer_v2_B
# local_configs.SUNRGBD.DFormer_v2_L