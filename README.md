# Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment

This repository contains the code of RoSTE introduced in our work: ["Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"]()

## Installation

Install Python environment
```bash
conda create -n verifier_env python=3.11 -y
source activate verifier_env

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Install Java environment (if needed)
```bash
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
```

You can run `bash quick_start_install.sh` for a quick start installation.


## Usage





## Acknowledgement

Our code implementation is built upon the open-source project [Verifiers](https://github.com/willccbb/verifiers).

## Citation

If you find our work useful in your research please consider citing our paper:
```

```