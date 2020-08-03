all: docker

conda-env: 
	conda env update -f env.yml

docker:
	# docker build -t pytorch-dist .
	# docker run --runtime=nvidia pytorch-dist 
	# -it --entrypoint bash
	docker-compose up --build

imagenet-dist:
	bash run-imagenet-dist.sh

docker-machine-create:
	docker-machine create \
		--driver amazonec2 \
		--amazonec2-instance-type m5.xlarge \
		--amazonec2-ami ami-049fb1ea198d189d7 \
		--amazonec2-region eu-central-1 \
		--amazonec2-zone b \
		--amazonec2-ssh-user ubuntu \
		--amazonec2-root-size 250 \
		node-$$ID-distributed-training