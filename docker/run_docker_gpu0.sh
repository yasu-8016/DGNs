#!/bin/bash

sudo docker run --gpus '"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
--memory-reservation 12gb \
--rm -it -u $(id -u):$(id -u) \
-v /data:/data \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
chem_py38:1.1.0 bash

