# repo

# Anonymous Submission: LiSenNet (Anonymous for Review)

This repository contains the official implementation of **LEN-NET**, a lightweight and effective speech enhancement network designed for real-time deployment on edge devices.

> 📄 This code is released as part of an anonymous submission to a peer-reviewed conference. Author and affiliation information has been removed for double-blind review.

---

🚀 Overview

LEN-NET consists of:
- A **Frequency-Time Distortion Balanced Encoder (FTMix)** for rich time-frequency feature extraction.
- A **Recurrent Dual-path Linear Transformer (RDL)** block for efficient long-range modeling.
- A **可微分pesq损失** 去替换判别器损失以减小训练消耗.

It achieves strong perceptual performance with low computational cost.

---

📦 Installation

Create a virtual environment and activate it:

```bash
git clone https://anonymous.4open.science/r/repo-9BE0.git
cd repo
conda create -n SEN python=3.7
conda activate SEN
pip install -r requirements.txt
```

📥 Data preparation

Download and extract the VoiceBank+DEMAND dataset. Resample all wav files to 16kHz, and move the [clean and noisy wavs](https://datashare.ed.ac.uk/handle/10283/1942) to /Datasets/wavs_clean and /Datasets/wavs_noisy, the test wavs to /Datasets/test_clean and /Datasets/test_noisy. 

respectively. You can also directly download the downsampled [16kHz dataset](https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l)(⚠️notice: Using this requires manually selecting two speakers as the test set.)


🏋️ Training
Before training, edit the configuration file ./config.yaml for your experiment.

Then run:

```bash
python train.py --config ./config.yaml
```
Training logs and checkpoints will be saved under /log.

🎧 Inference
```bash
python test.py --config ./config.yaml --ckpt_path path/to/checkpoint.ckpt --save_enhanced path/to/savedir
```

🧠 Model Architecture
Our model is composed of three main parts:

📁 Project Structure
```
├── Datasets/             # VoiceBank+DEMAND dataset
├── models/               # Model architecture definitions
│   └── discriminator/    (Optional) Used for discriminator loss
│   └── DP/               # Differentiable PESQ loss
│   └── lts/               # Our model
│   └── model.py
├── log/                  # Train log
├── result/               # Inference result
├── train.py              # Training entry point
├── test.py               # Evaluation script
├── config.yaml           # Main configuration
├── data_module.py        # Dataset loading and preprocessing
├── requirements.txt
└── README.md
```
🔊Samples


📄 License
This project is released for academic use only.

💬 Contact
Please reach out through the submission item if you have questions or suggestions.
