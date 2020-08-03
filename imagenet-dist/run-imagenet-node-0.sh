eval $(docker-machine env mo) && \
    docker-compose -f docker-compose.imagenet-dist.yml build && \
    docker-compose -f docker-compose.imagenet-dist.yml run --service-ports -e NODE_RANK=0 imagenet-dist
