#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python code/classification/train.py -c classification/config/E-ResNet18-tiny-imagenet.txt
# CUDA_VISIBLE_DEVICES=$1 python code/classification/train.py -c classification/config/E-ResNet18-cifar.txt