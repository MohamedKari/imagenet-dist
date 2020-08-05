# run distributed
awsenv
eval $(docker-machine env node-2-distributed-training)
bash image
bash imagenet-dist/run-imagenet-node-0.sh

```sh
python -m torch.distributed.launch \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr localhost \
    --master_port 8888 \
    toy_example.py
```

I use the above command to start the `torch.distributed.launch` utility inside the cantainer. Note that it assumes that we are processing 

The module will do the follwing for us: 

It starts up the script `nproc_per_node` times as a process [see GitHub](https://github.com/pytorch/pytorch/blob/91c80d122ab271d36ce37d60acf430fdbd54d249/torch/distributed/launch.py#L224) and, for each process, 
- sets the env variable `MASTER_ADDR` to the indicated value.
- sets the env variable `MASTER_PORT` to the indicated value.
- sets the env variable `WORLD_SIZE` to `nnodes * nproc_per_node` [see GitHub](https://github.com/pytorch/pytorch/blob/91c80d122ab271d36ce37d60acf430fdbd54d249/torch/distributed/launch.py#L205)
- sets the env variable `LOCAL_RANK` to one of `{0, 1, 2, ..., nproc_per_node - 1}`, so that each started process has a node-locally unique ID.
- sets the env variable `RANK` to `nproc_per_node * node_rank + local_rank`, so that each started process has a globally unqiue ID (i. e. across all machines involved) [see GitHub](https://github.com/pytorch/pytorch/blob/91c80d122ab271d36ce37d60acf430fdbd54d249/torch/distributed/launch.py#L226). This is where the assumption of equal and constant device counts across machines manifests. The TorchElastic launcher offers a away out by means of etcd as a "Rendez-vous server". 
- and also passes the the local rank (redundantly to the env variable) as an argument by setting `--local_rank=$LOCAL_RANK`.

E. g. calling 

```
python -m torch.distributed.launch \
    --nnodes 5 \
    --node_rank 3 \
    --nproc_per_node 4 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    toy_example.py
```

is equivalent to calling

```
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=20 LOCAL_RANK=0 RANK=12 toy_example.py --local_rank=0
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=20 LOCAL_RANK=1 RANK=13 toy_example.py --local_rank=0
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=20 LOCAL_RANK=2 RANK=14 toy_example.py --local_rank=0
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=20 LOCAL_RANK=3 RANK=15 toy_example.py --local_rank=0
```

The module doesn't _require_ any parameters because `nnodes` defaults to 1, `node_rank` to 0, `nproc_per_node` to 1, `master_addr` to 127.0.0.1 and  to 29500. Ergo, if no arguments are passed, the launch utility will start the training script once with `WORLD_SIZE=1`, `RANK=1` and `LOCAL_RANK=0`, thus allowing us to specialize to the case of a single-GPU training.

The moduel


https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py