import argparse
import sys
import yaml
import pytorch_lightning as pl
from pathlib import Path
import time
import torch

from model import Model
from data_module import DataModule


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.save_enhanced is not None:
        config['save_enhanced'] = args.save_enhanced
        Path(args.save_enhanced).mkdir(parents=True, exist_ok=True)
    
    model = Model(config=config)
    data_module = DataModule(**config['dataset_config'])
        # 3. 计算测试集总时长
    data_module.setup(stage='test')
    total_duration = 0.0
    sr = 16000  # 采样率为16000Hz
    for waveform, _, len, _ in data_module.test_dataloader():
        total_duration += len.cpu().numpy() / sr
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        logger=False,
    )

    t0 = time.perf_counter()
    trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    t1 = time.perf_counter()
    total_infer = t1 - t0

    # 5. 打印 RTF
    rtf = total_infer / total_duration
    print(f"—— 推理总耗时: {total_infer}s, 音频总时长: {total_duration}s, RTF: {rtf} ——")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test model')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_enhanced', type=str, default=None, help='The dir path to save enhanced wavs.')

    args = parser.parse_args()
    sys.exit(main(args))
