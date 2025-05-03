#!/bin/bash

# install python env
conda create -n verifier_env python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verifier_env

pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# install java env
apt-get remove --purge openjdk-21-jdk -y
apt-get update
apt-get install -y openjdk-21-jdk

# verify
java -version
