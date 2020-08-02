FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY env.yml env.yml
RUN conda env update -f env.yml --name base
COPY toy_example.py toy_example.py

ENV NNODES "<set-in-docker-compose-yml>"
ENV NODE_RANK "<set-in-docker-compose-yml>"
ENV NPROC_PER_NODE "<set-in-docker-compose-yml>"

ENTRYPOINT python -m torch.distributed.launch \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr localhost \
    --master_port 8888 \
    toy_example.py