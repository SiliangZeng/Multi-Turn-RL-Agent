# Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment

This repository contains the code of RoSTE introduced in our work: ["Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"]()

## Installation

Install Python environment
```bash
conda create -n verifier_env python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verifier_env

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Install Java environment (if needed)
```bash
apt-get remove --purge openjdk*

# Add Java 21 repository
apt-get remove --purge openjdk-21-jdk -y
apt-get update
apt-get install -y openjdk-21-jdk

# verify
java -version
```

You can run `bash quick_start_install.sh` for a quick start installation.

Download Wiki search index 
```bash
python verifiers/tools/local_wiki_search.py
```

## Usage





## Acknowledgement

Our code implementation is built upon the open-source project [Verifiers](https://github.com/willccbb/verifiers).

## Citation

If you find our work useful in your research please consider citing our paper:
```

```