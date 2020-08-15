#!/bin/bash

# Not sure what this does?
nvidia-smi

# Activate desired environment
# Need to source bashrc for some reason
source ~/.bashrc
conda activate pytorch-env

# Check to see if data is in the right spot
if [[ ! -d "/data/pacole2" ]]
then
    mkdir /data/pacole2
fi

if [[ ! -d "/data/pacole2/DeepLesionTestPreprocessed/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    cp /shared/rsaas/pacole2/DeepLesionTestPreprocessed.zip /data/pacole2/

    cd /data/pacole2/
    unzip -qq DeepLesionTestPreprocessed.zip
    rm DeepLesionTestPreprocessed.zip
    cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/DnCNN-PyTorch/
echo "Running python script now"
python -u test.py --data /data/pacole2/DeepLesionTestPreprocessed/miniStudies/ --logdir logs/DL/
