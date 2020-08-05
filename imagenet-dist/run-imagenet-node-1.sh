eval $(docker-machine env node-2-distributed-training) && \
    docker-compose -f docker-compose.imagenet-dist.yml build && \
    docker-compose -f docker-compose.imagenet-dist.yml run --detach --service-ports -e NODE_RANK=1 imagenet-dist
