mkdir -p ~/share/imagenet-distributed/data/
cd ~/share/imagenet-distributed/data/

aws s3 sync s3://imagenet-mo/chunked/ILSVRC2012_img_train/ train
aws s3 sync s3://imagenet-mo/chunked/ILSVRC2012_img_val/ val
find train -mindepth 1 -maxdepth 1 -name "*.tar" -type f -exec bash -c "tar -xf '{}' -C train --one-top-level; rm '{}'" \;
find val -mindepth 1 -maxdepth 1 -name "*.tar" -type f -exec bash -c "tar -xf '{}' -C val --one-top-level; rm '{}'" \;