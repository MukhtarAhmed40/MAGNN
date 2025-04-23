# MAGNN: Multi-scale Adaptive Graph Neural Networks for Malicious Traffic Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

Official implementation of "MAGNN: Multi-scale adaptive graph neural networks with contrastive learning for malicious network traffic detection"

![MAGNN Architecture](docs/magnn_architecture.png)

## Key Features

- 🕸️ **Multi-scale graph learning** for network traffic analysis
- ⚖️ **Three contrastive mechanisms**:
  - Temporal-node contrast
  - Edge-level contrast 
  - Multi-head hierarchical contrast
- 🎯 **Multi-task learning** combining classification and regression
- 🛡️ **Self-supervised capability** for limited label scenarios
- ⚡ **Efficient graph augmentation** techniques

## Table of Contents

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Usage](#usage)
4. [Metrics](#metrics)
5. [Results](#results)  
6. [Citation](#citation)
7. [License](#license)

## Installation

```bash
git clone https://github.com/yourusername/MAGNN.git
cd MAGNN

# Create conda environment (recommended)
conda create -n magnn python=3.8
conda activate magnn

# Install requirements
pip install -r requirements.txt

-----------------------------------------------------------------------------------------

## Requirements:

Python 3.8+

PyTorch 1.10+

PyTorch Geometric

Scikit-learn

TQDM

Matplotlib
----------------------------------------------------------------------------------
## Datasets
MAGNN supports these benchmark datasets:
___________________________________________________________________________
Dataset	- Type	- Nodes- 	Edges -	Description
CTU-13	- Botnet	- IPs -	Flows -	Botnet traffic scenarios
ISCXVPN2016	- VPN	- Hosts	- Sessions -	Encrypted VPN traffic
CICIDS2017 -	Intrusion	- IPs	- Connections	- Modern attack scenarios
CIRA-CIC-DoHBrw-2020	- DoH	- Domains	- Queries	D- NS over HTTPS traffic
__________________________________________________________________________
================================================================================
## Preprocessing:

python data/preprocess.py --dataset CTU-13 --output processed/ctu13

## Usage

# Training

python train.py \
  --config configs/ctu13.yaml \
  --data_dir processed/ctu13 \
  --output_dir models/

------------------------------------------------------

## Configuration options (see configs/base.yaml):

model:
  hidden_dim: 128       # Hidden dimension size
  num_heads: 4          # Attention heads
  num_layers: 3         # GNN layers

training:
  epochs: 100           # Training epochs
  batch_size: 64        # Batch size
  lr: 0.001             # Learning rate
  weight_decay: 1e-5    # L2 regularization

contrastive:
  temp: 0.1             # Temperature parameter
  augment_ratio: 0.2    # Edge modification ratio

---------------------------------------------------------------
## Evaluation

python evaluate.py \
  --model models/magnn_ctu13.pth \
  --data processed/ctu13/test \
  --output results.json
==================================================================






