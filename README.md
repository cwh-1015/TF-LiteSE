# Anonymous Submission: LTS-NET:A Lightweight Time-frequency Domain Speech Enhancement Network.(Anonymous for Review)

Abstract

> 📄 The goal of speech enhancement is to recover the ideal clean‐speech components from a noisy signal, which not only significantly improves listening comfort in real‐world noisy environments but also provides more reliable inputs for downstream tasks such as automatic speech recognition and speaker identification. However, mobile and IoT devices are often constrained in terms of computation, memory, and power, making the deployment of large‐scale deep neural networks impractical. To address this challenge, we propose **LTS-SEN**, a lightweight time-frequency domain speech enhancement network that achieves real‐time inference while maintaining competitive performance. Our design includes a global time–frequency downsampling module and a dual‐path time–frequency recurrent module, which work in concert to capture global time–frequency features while preserving low‐frequency details. With only 31K parameters, our model achieves a PESQ score of 3.21 on the VoiceBank--DEMAND dataset and a real‐time factor of RTF=0.013, demonstrating its practicality and generalizability in resource‐constrained environments.
---
⚠️Notice
> This code is released as part of an anonymous submission to a peer-reviewed conference. Author and affiliation information has been removed for double-blind review.
---

🚀 Overview

LTS-NET consists of:
- A **Frequency-Time Distortion Balanced Encoder (FTMix)** for rich time-frequency feature extraction.
- A **Recurrent Dual-path Linear Transformer (RDL)** block for efficient long-range modeling.
- A **Differentiable PESQ loss(DP)** replaces the discriminator loss to reduce training computational cost.

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

Respectively. You can also directly download the downsampled [16kHz dataset](https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l)(⚠️notice: Using this requires manually selecting two speakers as the test set.)


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

Reference
bash
[LiSenNet](https://github.com/hyyan2k/LiSenNet)
