# FuseKit
A robust transformer model testbench for proprietary and open-source models
* Intelligent multi-GPU inference and training
* API model inference
* Powerful multimodal dataset implementation
* Built-in LLM-As-A-Judge similarity metric
* Extends PEFT to enable inference-time model composition

# Getting Started
1. Create a clean Python 3.11 environment
2. Install Pytorch `pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. Install FuseKit `pip install --no-cache-dir fusekit`
4. Initialize FuseKit with `fusekit init`
5. Export FUSEKIT_MODELS FUSEKIT_APIKEYS to point to folders with local models and API keys for models, or modify the config.yml
    1. Filepaths for models and apikeys can be found in fusekit.Common.env
    2. *.apikey files should only contain the plaintext key; however, they can use # to comment out entire lines
    3. *.org files re required, but if you an org api key, then leave this file blank


For funding acknowledgments and required U.S. Government disclaimers, see:
[Funding & Disclaimer](./DISCLAIMER.md)