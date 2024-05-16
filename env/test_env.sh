set -ex
# driver version 535.161.08
cat /proc/driver/nvidia/version

# cuda version cuda_11.0_bu.TC445_37.28845127_0
nvcc --version

# cudnn version
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 0
#define CUDNN_PATCHLEVEL 5
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# TRT version
# cat /path/to/TRT/TensorRT-8.6.1.6/include/NvInferVersion.h | grep NV_TENSORRT_MAJOR -A 3

# mkl mirror
# https://mirror.nju.edu.cn/anaconda/cloud/intel/

if [ ! -d "bin" ]; then
    mkdir bin
fi
# test cuda
nvcc -o bin/test_cuda test/test_cuda.cu
./bin/test_cuda

# test cudnn
nvcc test/test_cudnn.cu -lcudnn -o bin/test_cudnn
./bin/test_cudnn

# test tensorRT
# path/to/TRT/samples/sampleOnnxMNIST

echo "Success!"