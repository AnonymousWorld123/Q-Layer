# pixelcnn_keras #
#### This is keras implementation of Gated PixelCNN model as proposed in [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328).
- Plese let me know if you find any issues or bugs:
https://github.com/suga93/pixelcnn_keras/issues



## Requirements ##

### Python ###
- Version: 2.7.X or 3.4.X

### TensorFlow ###
- Version: 1.0.0rc1
- GPU: enabled

### Keras ###
- Version: 1.2.2

### other packages ###
- graphviz

### other python dependencies ###
- scipy==0.18.1
- numpy==1.12.0
- scikit-image==0.12.3
- h5py==2.6.0
- pydot-ng==1.0.0



## Setup with Docker ##
This repository contains Dockerfile of execution environment. Docker image is registered using [automated build](https://docs.docker.com/docker-hub/builds/).

### Base Docker Image ###
* [tensorflow/tensorflow:1.0.0-rc1-devel-gpu-py3] (https://hub.docker.com/r/tensorflow/tensorflow/)

### My Docker Image ###
* [suga93/pixelcnn_keras](https://hub.docker.com/r/suga93/pixelcnn_keras/)

### Requirements ###
- Linux OS
- GPU
- NVIDIA Driver
- Docker
- NVIDIA-Docker

### Installation ###
1. Install [Docker](https://www.docker.com/)
2. Install [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker)
3. Get docker image: `docker pull suga93/pixelcnn_keras`
4. Run docker image: `nvidia-docker run -it suga93/pixelcnn_keras`



## Usage ##

### Train (conditional) Gated PixelCNN model using keras datasets ###
#### Show help message: ####
	python3 train_keras_datasets.py -h
#### Train unconditional sample mnist model (set config options): ####
	python3 train_keras_datasets.py -c configs/sample_train_mnist_small.cfg
#### Train mnist model with overwriting some config options: (ex. unconditional -> conditional) ####
	python3 train_keras_datasets.py -c configs/sample_train_mnist_small.cfg --conditional True
#### Train cifar10 model: ####
	python3 train_keras_datasets.py -c configs/sample_train_cifar10.cfg

### Generate images from trained model ###
* In this implementation, Keras fails to save model architecture (model.save(), model.to_json(), model.to_yaml()). Therefore, first you need to build same architecture as your trained model.

#### Show help message: ####
	python3 predict_keras_datasets.py -h
#### Generate mnist images from trained model: ####
	python3 predict_keras_datasets.py -c configs/sample_predict_mnist.cfg --checkpoint_file /path/to/trained_model_mnist.hdf5
#### Generate cifar10 images from trained model: ####
	python3 predict_keras_datasets.py -c configs/sample_predict_cifar10.cfg --checkpoint_file /path/to/trained_model_cifar10.hdf5

