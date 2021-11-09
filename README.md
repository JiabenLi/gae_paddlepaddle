# gae_paddlepaddle
Variational Graph Auto-Encoders in PaddlePaddle

This is a PaddlePaddle implementation of the Variational Graph Auto-Encoder model described in the paper:

T. N. Kipf, M. Welling, Variational Graph Auto-Encoders, NIPS Workshop on Bayesian Deep Learning (2016)

The code in this repo is based on or refers to https://github.com/zfjsail/gae-pytorch

# Requirements
* Python 3
* PaddlePaddle-2.1.3
* install requirements via pip install -r requirements.txt

# Usage
train: python gae/train.py --data_path "your_path", for example: /lustre/S/fuqiang/gae_paddlepaddle/gae/data

eval: python gae/eval.py --data_path "your_path", for example: /lustre/S/fuqiang/gae_paddlepaddle/gae/data
