# Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design 

This repository contains Tensorflow implementation of experiments from the paper [Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design - Jonathan Ho*, Xi Chen*, Aravind Srinivas, Yan Duan, Pieter Abbeel](https://arxiv.org/abs/1902.00275)

# Dependencies

* python3.6 
* Tensorflow v1.10.1 
* horovod v0.14.1

# Usage Instructions

We trained our models using 8 GPUs with data-parallelism. 

# CIFAR 10 
```
mpiexec -n 8 python -m flows.run_cifar
```
# Imagenet 32

Data: Script to create dataset [here](https://github.com/aravind0706/flowpp/blob/master/flows_imagenet/create_imagenet_benchmark_datasets.py)

```
mpiexec -n 8 -m flows_imagenet.launchers.imagenet32_official
```
# Imagenet 64
```
mpiexec -n 6 python3.6 -m flows_imagenet.launchers.imagenet64_official
```
# CelebA-HQ 64x64 

Data: Download links in [README](https://github.com/aravind0706/flowpp/tree/master/flows_celeba)

```
mpiexec -n 8 python3.6 -m flows_celeba.launchers.celeba64_5bit_official
```
# Contact

Please open an issue

# Credits

flowpp was originally developed by Jonathan Ho (UC Berkeley), Peter Chen (UC Berkeley / covariant.ai), Aravind Srinivas (UC Berkeley), Yan Duan (covariant.ai), and Pieter Abbeel (UC Berkeley / covariant.ai). 
