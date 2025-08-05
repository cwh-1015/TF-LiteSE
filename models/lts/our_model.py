import torch
import torch.nn as nn
from .dpl import CustomLayerNorm, DPL

import torch.nn.functional as F
import os
import math

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class FTMix(nn.Module):
    def __init__(self, in_c, out_c, k_f=5, stride_f=2, k_t=3):
        super().__init__()

        self.freq_conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=(1, k_f),
            stride=(1, stride_f),
            padding=(0, k_f // 2),  
            bias=False
        )

        self.time_conv = nn.Conv2d(
            out_c, out_c,
            kernel_size=(k_t, 1),
            padding=(k_t // 2, 0),
            groups=out_c,
            bias=False
        )

    def forward(self, x):             # (B,C,T,257)
        x = self.freq_conv(x)         
        x = self.time_conv(x)
        return x


class Encoder2Step(nn.Module):
    def __init__(self, in_c=3, hid_c=16):
        super().__init__()
        self.ch_expand = nn.Sequential(
            nn.Conv2d(in_c, hid_c, 1, 1),
            CustomLayerNorm((1, 256), stat_dims=(1, 3)),
            nn.PReLU(hid_c)
        )
        
        self.ft1 = FTMix(hid_c, hid_c, stride_f=2)
        self.ft2 = FTMix(hid_c, hid_c, stride_f=2)

    def forward(self, x):
        x = x[..., :-1]
        x = self.ch_expand(x)   # (B,16,T,257)
        x = self.ft1(x)         # (B,16,T,128)
        x = self.ft2(x)         # (B,16,T, 64)
        return x 

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, n_freqs, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.0)
        self.num_channels = in_channels//4
        self.conv = nn.Conv2d(
            in_channels, self.num_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(self.num_channels, out_channel, (2, 2)), # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(n_freqs, beta=1)

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        x = self.mask_conv(out)  # (B,out_channel,T,F)
        x = x.permute(0, 3, 2, 1)  # (B,F,T,out_channel)
        x = self.lsigmoid(x).permute(0, 3, 2, 1)
        return x    

class FTPerceptNet(nn.Module):
    def __init__(self, num_channels=16, out_channel=2, n_blocks=2, n_fft=512, hop_length=256, compress_factor=0.3):
        super(FTPerceptNet, self).__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.compress_factor = compress_factor
        self.enc   = Encoder2Step(in_c=3, hid_c=num_channels)   # 257→64
        self.proc = nn.Sequential(*[DPL(emb_dim=num_channels, hidden_dim=num_channels*3//2, n_freqs=64, dropout_p=0.1,) for _ in range(n_blocks)])
        self.dec   = SPConvTranspose2d(num_channels, out_channel=2, kernel_size=(1,3), n_freqs=self.n_freqs, r=4)
        self.pad1  = nn.ConstantPad2d((0,1,0,0), 0.0)          # 256→257
        

    def apply_stft(self, x, return_complex=True):
        # x:(B,T)
        assert x.ndim == 2
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)  # (B,T,F)
        return spec

    def apply_istft(self, x, length=None):
        # x:(B,T,F)
        assert x.ndim == 3
        x = x.transpose(1, 2)  # (B,F,T)
        audio = torch.istft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            length=length,
            return_complex=False
        )  # (B,T)
        return audio

    def power_compress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** self.compress_factor
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def power_uncompress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** (1.0 / self.compress_factor)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    
    def mel_scale(self, mag, sr=16000, f_min=0.0, f_max=8000.0, n_mels=64):
        if not hasattr(self, 'fb'):
            fb = melscale_fbanks(n_freqs=self.n_freqs, f_min=f_min, f_max=f_max, n_mels=n_mels, sample_rate=sr)
            setattr(self, 'fb', fb.to(mag.device))
        mag = mag ** (1 / self.compress_factor)  
        mel = mag @ self.fb  
        mel = mel ** self.compress_factor  
        return mel

    @staticmethod
    def cal_gd(x):
        b, t, f = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(b, t, 1, device=x.device))  
        return torch.atan2(x_gd.sin(), x_gd.cos())

    @staticmethod
    def cal_if(x):
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  
        return torch.atan2(x_if.sin(), x_if.cos())
    
    def cal_ifd(self, x):
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  
        x_ifd = x_if - 2 * torch.pi * (self.hop_length / self.n_fft) * torch.arange(f, device=x.device)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def griffinlim(self, mag, pha=None, length=None, n_iter=2, momentum=0.99):
        mag = mag.detach()
        mag = mag ** (1.0 / self.compress_factor) 
        assert 0 <= momentum < 1
        momentum = momentum / (1 + momentum)
        if pha is None:
            pha = torch.rand(mag.size(), dtype=mag.dtype, device=mag.device)

        tprev = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)
        for _ in range(n_iter):
            inverse = self.apply_istft(torch.complex(mag * pha.cos(), mag * pha.sin()), length=length)
            rebuilt = self.apply_stft(inverse)
            pha = rebuilt
            pha = pha - tprev.mul_(momentum)
            pha = pha.angle()
            tprev = rebuilt

        return pha

    def forward(self, src, tgt=None):              

        if tgt == None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))  # (B,T,F)
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)

        tgt_spec = self.power_compress(self.apply_stft(tgt))  # (B,T,F)
        tgt_mag = tgt_spec.abs()

        x = torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)  # (B,3,T,F)
        x = self.enc(x)             # (B,16,T,64)
        copy = x

        x = self.proc(x)            # same sixe
        z = x + copy

        x = self.dec(z)             # (B,2,T,257)


        est_mag = (x[:, 0] + 1e-8) * src_mag + (x[:, 1] + 1e-8) * src_mag

        est_pha = self.griffinlim(est_mag.detach(), src_pha, tgt.size(-1))
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est = self.apply_istft(self.power_uncompress(est_spec), length=tgt.size(-1))

        results = {
            'tgt': tgt,
            'tgt_spec': tgt_spec,
            'tgt_mag': tgt_mag,
            'est': est,
            'est_spec': est_spec,
            'est_mag': est_mag,
        }

        return results