FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY env.yml env.yml
RUN conda env update -f env.yml --name base
COPY imagenet-dist/imagenet_main.py imagenet_main.py



# NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL 
# ENTRYPOINT python imagenet_main.py /share/data -a resnet18 --workers 2 --print-freq 1 --checkpoint-file /share/checkpoints/
ENTRYPOINT python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --use_env \
    imagenet_main.py \
    --arch resnet18 \
    --epochs 60 \
    --batch-size 32 \
    --workers 4 \
    --lr 0.1 \
    --print-freq 10 \
    --checkpoint-file /share/checkpoints/checkpoint.pth.tar \
    /share/data
