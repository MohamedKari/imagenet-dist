eval $(docker-machine env mo) && \
    docker-compose -f docker-compose.imagenet-dist.yml build && \
    docker-compose -f docker-compose.imagenet-dist.yml run -e NODE_RANK=0 imagenet-dist

eval $(docker-machine env node-2-distributed-training) && \
    docker-compose -f docker-compose.imagenet-dist.yml build && \
    docker-compose -f docker-compose.imagenet-dist.yml run -e NODE_RANK=1 imagenet-dist