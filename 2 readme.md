# MAGNN: Multi-Scale Adaptive Graph Neural Networks for Malicious Traffic Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

Official implementation of the paper:

**"MAGNN: Multi-Scale Adaptive Graph Neural Networks with Contrastive Learning for Malicious Network Traffic Detection"**

![MAGNN Architecture](docs/magnn_architecture.png)

---

## 🔑 Key Features

- 🕸️ **Multi-scale graph learning** for malicious network traffic detection
- ⚖️ **Three contrastive mechanisms**:
  - Temporal-node contrast
  - Edge-level contrast
  - Multi-head hierarchical contrast
- 🎯 **Multi-task learning**: Joint classification and temporal prediction
- 🛡️ **Self-supervised training**: Robust to scarce labeled data
- ⚡ **Graph augmentations**: Edge modification + Random Walk with Restart (RWR)

---

## 📚 Table of Contents

1. [Installation](#installation)
2. [Requirements](#requirements)
3. [Datasets](#datasets)
4. [Preprocessing](#preprocessing)
5. [Usage](#usage)
6. [Evaluation](#evaluation)
7. [Citation](#citation)
8. [License](#license)
9. [Contact & Support](#contact--support)

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/MAGNN.git
cd MAGNN

# (Recommended) Create a virtual environment
conda create -n magnn python=3.8
conda activate magnn

# Install required packages
pip install -r requirements.txt

✅ Requirements
Python 3.8+

PyTorch ≥ 1.10

PyTorch Geometric

scikit-learn

matplotlib

pandas, numpy

Jupyter Notebook or Google Colab (optional)

📂 Datasets
MAGNN supports four benchmark datasets:


Dataset	Type	Nodes	Edges	Description
CTU-13	Botnet	IPs	Flows	13 scenarios of botnet traffic
ISCXVPN2016	VPN	Hosts	Sessions	Encrypted VPN and non-VPN sessions
CICIDS-2017	Intrusion	IPs	Connections	Modern day network attacks
CIRA-CIC-DoHBrw-2020	DNS/DoH	Domains	Queries	DNS over HTTPS (DoH) communication

🛠️ Preprocessing
python data/preprocess.py --dataset CTU-13 --output processed/ctu13

🧪 Usage
🔧 Training
python train.py \
  --config configs/ctu13.yaml \
  --data_dir processed/ctu13 \
  --output_dir models/

⚙️ Configuration (example from configs/base.yaml)

Model:
  hidden_dim: 128
  num_heads: 4
  num_layers: 3

Training:
  epochs: 100
  batch_size: 64
  lr: 0.001
  weight_decay: 1e-5

Contrastive:
  temp: 0.1
  augment_ratio: 0.2


🧾 Evaluation
python evaluate.py \
  --model models/magnn_ctu13.pth \
  --data processed/ctu13/test \
  --output results.json

 Evaluation Metrics
Classification: Accuracy, Precision, Recall, F1-Score

Regression/Prediction: MAE, MSE, RMSE, MAPE

📊 Results
MAGNN outperforms all baseline models across all datasets with up to:

+8.5% Accuracy

+7.3% F1-Score

−12.4% RMSE

Supports fast convergence and low inference latency under distributed training.

📄 License
This project is licensed under the MIT License.

📬 Contact & Support
For questions, feedback, or contributions:

Mukhtar Ahmed
📧 Email: mukhtar40@gmail.com
🔗 LinkedIn: Mukhtar Ahmed

