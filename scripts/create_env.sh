#!/bin/bash
# Install Anaconda3
#curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
#bash installer.sh -b -f && \
#source ~/.bashrc &&\
#rm installer.sh
#export PATH="/root/anaconda3/bin:$PATH" && \

conda update -y conda
conda create -y -n ctg python=3.7
conda activate ctg

#conda install -y -n ctg pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.1 -c pytorch
conda install -y -n ctg pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch

pip install --upgrade pip
pip install -r requirements.txt