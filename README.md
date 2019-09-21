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
