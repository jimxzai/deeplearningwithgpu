# deeplearningwithgpu
environment standard for our deep learning with GPU support

supported environment:
. Ubuntu 16.04 LTS / Ubuntu 18.04 LTS (with driver from Nvidia  435.21)
. CUDA 10.1 (update 2)
. Conda (need some manual tweaking to make it work with CUDA 10.1 instead of 10.0)
. Python 3.7 
. Tensorflow 2.x RC
. GPU verfied (RTX 2080Ti, GTX 1070, GTX 1060)
. Alienware 15 R3, R4 ; Alienware m15


Things to tweak:
1. for ubuntu 18.04 needs to be careful as the recent release of driver 430 will fail the kernel 62. 435 fixed it;
you may need to purge all NVIDIA driver and reinstall with 435.

2. for dell alienware, BIOS needs to be updated to the recent version to load Linux (ubuntu/centos), secure boot needs to be disabled in BIOS;

3. CUDA version with TF version confliction ( manual tweaking)

Another alternative is to use Docker
https://www.tensorflow.org/install/docker


 Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker


#### Test nvidia-smi with the latest official CUDA image
docker run --gpus all nvidia/cuda:9.0-base nvidia-smi
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

# Start a GPU enabled container on two GPUs
$ docker run --gpus 2 nvidia/cuda:9.0-base nvidia-smi

# Starting a GPU enabled container on specific GPUs
$ docker run --gpus '"device=1,2"' nvidia/cuda:9.0-base nvidia-smi
$ docker run --gpus '"device=UUID-ABCDEF,1'" nvidia/cuda:9.0-base nvidia-smi

# Specifying a capability (graphics, compute, ...) for my container
# Note this is rarely if ever used this way
$ docker run --gpus all,capabilities=utility nvidia/cuda:9.0-base nvidia-smi

docker pull tensorflow/tensorflow                     # latest stable release
docker pull tensorflow/tensorflow:devel-gpu           # nightly dev release w/ GPU support
docker pull tensorflow/tensorflow:latest-gpu-jupyter  # latest release w/ GPU support and Jupyter


http://localhost:8889/?token=a4460bc17dfb20cfa1ec196d4feb407de6c9ab5e1cdbd265


steps for ubuntu 18.04 deep learning
1. nvidia driver 435
2. cuda 10.1

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

