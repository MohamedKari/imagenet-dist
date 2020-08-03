FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY env.yml env.yml
RUN conda env update -f env.yml --name base
COPY toy_example.py toy_example.py

# NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL
ENTRYPOINT python -m torch.distributed.launch \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    toy_example.py

# ENTRYPOINT python -m http.server --bind 0.0.0.0 8080