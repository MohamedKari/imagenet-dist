eval $(docker-machine env node-2-distributed-training) && docker-compose run --service-ports -e NODE_RANK=0 pytorch-dist