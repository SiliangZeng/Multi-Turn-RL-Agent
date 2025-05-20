#!/bin/bash

# install python env
conda create -n verifier_env python=3.11 -y
source activate verifier_env

pip install -e .
pip install flash-attn --no-build-isolation

# install java env
apt-get remove --purge openjdk* -y
apt-get update
apt-get install -y openjdk-21-jdk

# verify
java -version
