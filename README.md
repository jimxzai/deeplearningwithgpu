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


verification for Tensorflow 2.0

import tensorflow as tf;
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

you should see the GPU being loaded and use nvidia-smi you will see the GPU memory and the PID from python.



# Keras-TensorFlow-GPU-Windows-Installation (Updated: 12th Apr, 2019)
## 10 easy steps on the installation of TensorFlow-GPU and Keras in Windows

### Step 1: Install NVIDIA Driver <a href="https://www.nvidia.com/Download/index.aspx?lang=en-us" target="_blank">Download</a>
Select the appropriate version and click search
<p align="center"><img width=80% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/NVIDIA_Driver_installation_v2.png"></p>

### Step 2: Install Anaconda (Python 3.7 version) <a href="https://www.anaconda.com/download/" target="_blank">Download</a>
<p align="center"><img width=80% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/anaconda_windows_installation_v2.png"></p>

### Step 3: Update Anaconda
Open Anaconda Prompt to type the following command(s)
```Command Prompt
conda update conda
conda update --all
```

### Step 4: Install CUDA Tookit 10.0 <a href="https://developer.nvidia.com/cuda-downloads" target="_blank">Download</a>
Choose your version depending on your Operating System

<p align="center"><img width=90% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/cuda10_windows10_local_installation.png"></p>

### Step 5: Download cuDNN <a href="https://developer.nvidia.com/rdp/cudnn-download" target="_blank">Download</a>
Choose your version depending on your Operating System.
Membership registration is required.

<p align="center"><img width=90% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/cuDNN_windows_download_v2.png"></p>

Put your unzipped folder in C drive as follows: 
```Command Prompt
D:\cudnn-10.1-windows10-x64-v7.5.0.56
```
### Step 6: Add cuDNN into Environment PATH 

Add the following path in your Environment.
Subjected to changes in your installation path.
```Command Prompt
D:\cudnn-8.0-windows10-x64-v5.1\cuda\bin
```

You can either follow this <a href="https://kb.wisc.edu/cae/page.php?id=24500" target="_blank">Tutorial</a> here or the following steps (for Windows 10).

#### Step 6.1: open the Start Search, type in “env”
#### Step 6.2: choose “Edit environment variables for your account”:

<p align="center"><img width=80% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/add_environment_variable_01.png"></p>

#### Step 6.3: under the “Users' Variables” section (the upper half), find the row with “Path” in the first column, and click edit.
<p align="center"><img width=80% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/add_environment_variable_02.png"></p>

#### Step 6.4: the “Edit environment variable” user interface will appear. click New.

<p align="center"><img width=80% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/add_environment_variable_03.png"></p>

Now you can add a new path to the environment varible

Turn off all the prompts. 
Open a new Anaconda Prompt to type the following command(s)
```Command Prompt
echo %PATH%
```
You shall see that the new Environment PATH is there.

### Step 7: Create an Anaconda environment with Python=3.6
Open Anaconda Prompt to type the following command(s)
```Command Prompt
conda create -n keras-gpu python=3.6 numpy scipy keras-gpu
```

### Step 8: Activate the environment
Open Anaconda Prompt to type the following command(s)
```Command Prompt
activate keras-gpu
```

### Step 9: Testing
Let's try running <a href="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/examples/mnist_mlp.py">mnist_mlp.py</a> in your prompt.

Open Anaconda Prompt to type the following command(s)
```Command Prompt
activate keras-gpu
python mnist_mlp.py
```
Congratulations ! You have successfully run Keras (with Tensorflow backend) over GPU on Windows !

<p align="left"><img width=100% src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/blob/master/installation_success_v2.png"></p>

### Step 10: Done !
