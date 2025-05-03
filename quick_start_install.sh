#!/bin/bash

# install python env
conda create -n verifier_env python=3.11 -y
source activate verifier_env

pip install -r requirements.txt
pip install flash-attn --no-build-isolation


# install jave env (if needed)
apt-get remove --purge openjdk*

# Add Java 21 repository
apt-get update
apt-get install -y wget gpg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list

# Install Java 21
apt-get update
apt-get install -y temurin-21-jdk

# Set Java environment variables
export PATH=$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server:$LD_LIBRARY_PATH
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
export JVM_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server/libjvm.so

# Verify Java installation
java -version
