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
Requirements:

Python 3.8+

PyTorch 1.10+

PyTorch Geometric

Scikit-learn

TQDM

Matplotlib

Datasets
MAGNN supports these benchmark datasets:

Dataset	Type	Nodes	Edges	Description
CTU-13	Botnet	IPs	Flows	Botnet traffic scenarios
ISCXVPN2016	VPN	Hosts	Sessions	Encrypted VPN traffic
CICIDS2017	Intrusion	IPs	Connections	Modern attack scenarios
CIRA-CIC-DoHBrw-2020	DoH	Domains	Queries	DNS over HTTPS traffic
Preprocessing:

bash
Copy
python data/preprocess.py --dataset CTU-13 --output processed/ctu13
Usage
Training
bash
Copy
python train.py \
  --config configs/ctu13.yaml \
  --data_dir processed/ctu13 \
  --output_dir models/
Configuration options (see configs/base.yaml):

yaml
Copy
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
Evaluation
bash
Copy
python evaluate.py \
  --model models/magnn_ctu13.pth \
  --data processed/ctu13/test \
  --output results.json
Metrics
MAGNN evaluates both classification and regression performance:

Classification Metrics
Metric	Formula	Description
Accuracy	(TP+TN)/(TP+TN+FP+FN)	Overall correctness
F1-Score	2(PrecisionRecall)/(Precision+Recall)	Harmonic mean of precision/recall
Error Metrics (Temporal Prediction)
Metric	Formula	Interpretation
MAE	
1
n
∑
∥
y
−
y
^
∥
n
1
​
 ∑∥y− 
y
^
​
 ∥	Average error magnitude
MSE	
1
n
∑
(
y
−
y
^
)
2
n
1
​
 ∑(y− 
y
^
​
 ) 
2
 	Squared error (penalizes large errors)
RMSE	
MSE
MSE
​
 	Error in original units
MAPE	
100
%
n
∑
∥
(
y
−
y
^
)
/
y
∥
n
100%
​
 ∑∥(y− 
y
^
​
 )/y∥	Percentage error
Results
Reproduction of paper results on CTU-13 dataset:

Metric	Value	Baseline Comparison
Accuracy	98.45%	+8.5% over Anomal-E
F1-Score	97.58%	+7.3% over NEGAT
RMSE	0.20	-12.4% reduction
Training Time	18.5s/epoch	2.1x faster than DE-GNN
Training Curves

Citation
If you use MAGNN in your research, please cite:

bibtex
Copy
@article{ahmed2024magnn,
  title={MAGNN: Multi-scale adaptive graph neural networks with contrastive learning for malicious network traffic detection},
  author={Ahmed, Mukhtar and Chen, Jinfu and Akpaku, Ernest and Bux, Ali},
  journal={Journal of Network and Computer Applications},
  year={2024},
  publisher={Elsevier}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.

Copy

### Key Features of This README:

1. **Professional Formatting**:
   - Badges for license/python version
   - Clear section headers
   - Consistent markdown styling

2. **Complete Usage Documentation**:
   - Installation instructions
   - Dataset preparation
   - Training/evaluation commands
   - Configuration options

3. **Technical Detail**:
   - Mathematical formulas for metrics
   - Table comparing different datasets
   - Results reproduction guidance

4. **Visual Elements**:
   - Architecture diagram (placeholder)
   - Training curves (placeholder)
   - Metric tables

5. **Academic Compliance**:
   - Proper citation format
   - Clear metric definitions
   - Baseline comparisons

To use this README:
1. Save as `README.md` in your project root
2. Add actual images to `docs/` folder
3. Update URLs and paths as needed
4. Modify results based on your actual experiments

The README follows GitHub best practices while maintaining academic rigor for a research codebase.



































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






