#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python code/classification/train.py -c classification/config/E-ResNet18.txt