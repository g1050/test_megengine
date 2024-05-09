nvidia-docker run -p 10831:10831 -it --runtime=nvidia -v /data:/data --name xkgao_cuda11 nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 /bin/bash
