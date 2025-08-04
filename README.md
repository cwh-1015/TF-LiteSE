# repo

# Anonymous Submission: LiSenNet (Anonymous for Review)

This repository contains the official implementation of **LEN-NET**, a lightweight and effective speech enhancement network designed for real-time deployment on edge devices.

> 📄 This code is released as part of an anonymous submission to a peer-reviewed conference. Author and affiliation information has been removed for double-blind review.

---

## 🚀 Overview

LEN-NET consists of:
- A **Frequency-Time Distortion Balanced Encoder (FTMix)** for rich time-frequency feature extraction.
- A **Recurrent Dual-path Linear Transformer (RDL)** block for efficient long-range modeling.
- A **可微分pesq损失** 去替换判别器损失以减小训练消耗.

It achieves strong perceptual performance with low computational cost.

---

## 📦 Installation

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:

bash
复制
编辑
pip install -r requirements.txt
✅ Tested with: Python==3.10.14, PyTorch==2.0.0, PyTorch Lightning==2.0.7

🏋️ Training
Before training, edit the configuration file ./config.yaml to set:

devices: e.g., [0] for GPU 0

logdir: path to save logs and checkpoints

data.train_dir: directory of training set

batch_size, learning_rate, max_epochs, etc.

Then run:

bash
python train.py --config ./config.yaml
Training logs and checkpoints will be saved under logdir.

🎧 Evaluation
To evaluate a trained checkpoint:

bash
python test.py --config ./config.yaml --ckpt_path path/to/checkpoint.ckpt
To also save enhanced audio samples:

bash
python test.py --config ./config.yaml --ckpt_path path/to/checkpoint.ckpt --save_enhanced ./enhanced_audio/
🔧 Configuration Example (config.yaml)
yaml
devices: [0]
logdir: ./exp/lisennet
data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
  sr: 16000
batch_size: 16
learning_rate: 1e-4
max_epochs: 100
num_workers: 4
🧠 Model Architecture
The network is composed of three main parts:

📁 Project Structure
bash
复制
编辑
├── train.py              # Training entry point
├── test.py               # Evaluation script
├── config.yaml           # Main configuration
├── models/               # Model architecture definitions
│   └── lisennet.py
├── data/                 # Dataset loading and preprocessing
├── utils/                # Helper functions
├── requirements.txt
└── README.md
🔒 Anonymous Review Note
This repository has been anonymized to comply with double-blind review policies. Please do not attempt to de-anonymize the authors.

📄 License
This project is released for academic use only.

💬 Contact
Please reach out through the submission system if you have questions or suggestions.
