import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.utils import calc_diffusion_step_embedding
from models.s4 import S4

class TransposedLN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1))
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, x):
        s, m = torch.std_mean(x, dim=-2, unbiased=False, keepdim=True)
        y = (self.s/s) * (x-m+self.m)
        return y

""" Pooling functions with trainable parameters """
class DownPool(nn.Module):
    def __init__(self, d_input, d_output, pool):
        super().__init__()
        self._d_output = d_output
        self.pool = pool

        # self.linear = nn.Conv1d(
        self.linear = Conv(
            d_input * pool,
            d_output,
            1,
        )

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x

class UpPool(nn.Module):
    def __init__(self, d_input, d_output, pool):
        super().__init__()
        self.d_input = d_input
        self._d_output = d_output
        self.pool = pool

        self.linear = Conv(
            d_input,
            d_output * pool,
            1,
        )

    def forward(self, x, *args, **kwargs):
        x = self.linear(x)
        # x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)
        return x

class FF(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        d_inner = expand * d_model

        linear1 = Conv(d_model, d_inner, 1)
        linear2 = Conv(d_inner, d_model, 1)

        self.ff = nn.Sequential(
            linear1,
            nn.GELU(),
            linear2,
        )

    def forward(self, x, *args, **kwargs):
        return self.ff(x)


def swish(x):
    return x * torch.sigmoid(x)


# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


# every residual block
# contains one noncausal dilated conv
class DiffWaveBlock(nn.Module):
    def __init__(self,
            d_model, L, ff,
            diffusion_step_embed_dim_out=512,
            unconditional=False,
            mel_upsample=[16,16],
        ):
        super().__init__()
        self.d_model = d_model

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.d_model)

        self.layer = S4(d_model, l_max=L, bidirectional=True)
        self.ff = FF(d_model, ff)

        self.norm1 = TransposedLN(d_model)
        self.norm2 = TransposedLN(d_model)

        self.unconditional = unconditional
        if not self.unconditional:
            # add mel spectrogram upsampler and conditioner conv1x1 layer
            self.upsample_conv2d = torch.nn.ModuleList()
            for s in mel_upsample:
                conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
                conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
                torch.nn.init.kaiming_normal_(conv_trans2d.weight)
                self.upsample_conv2d.append(conv_trans2d)
            self.mel_conv = Conv(80, self.d_model, kernel_size=1)  # 80 is mel bands

    def forward(self, x, diffusion_step_embed, mel_spec=None):
        y = x
        B, C, L = x.shape
        assert C == self.d_model

        y = self.norm1(y)

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        y = y + part_t.unsqueeze(-1)
        # part_t = part_t.view([B, self.d_model, 1]) # TODO should be unsqueeze?

        # y = self.norm1(y)
        # dilated conv layer
        y, _ = self.layer(y)

        # add mel spectrogram as (local) conditioner
        if mel_spec is not None:
            assert not self.unconditional
            # Upsample spectrogram to size of audio
            mel_spec = torch.unsqueeze(mel_spec, dim=1)
            mel_spec = F.leaky_relu(self.upsample_conv2d[0](mel_spec), 0.4)
            mel_spec = F.leaky_relu(self.upsample_conv2d[1](mel_spec), 0.4)
            mel_spec = torch.squeeze(mel_spec, dim=1)
            # print(mel_spec.shape)

            # print(mel_spec.size(2), L)
            assert(mel_spec.size(2) >= L)
            if mel_spec.size(2) > L:
                mel_spec = mel_spec[:, :, :L]

            mel_spec = self.mel_conv(mel_spec)
            y = y + mel_spec

        y = x + y

        x = y
        y = self.norm2(y)
        y = self.ff(y)
        y = x + y

        return y



class Sashimi(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
            d_model=64,
            n_layers=8,
            pool=[4,4],
            expand=2,
            ff=2,
            unet=True,
            diffusion_step_embed_dim_in=128,
            diffusion_step_embed_dim_mid=512,
            diffusion_step_embed_dim_out=512,
            unconditional=False,
            mel_upsample=[16,16],
            L=16000,
            **kwargs,
        ):
        super().__init__()

        self.L = L
        self.unet = unet
        self.d_model = d_model
        self.n_layers = n_layers
        self.expand = expand
        self.ff = ff
        self.pool = pool

        # initial conv1x1 with relu
        self.init_conv = nn.Sequential(Conv(in_channels, d_model, kernel_size=1), nn.ReLU())

        # self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # the shared two fc layers for diffusion step embedding
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        def _residual(d, L):
            return DiffWaveBlock(
                d, L, ff,
                unconditional=unconditional,
                mel_upsample=mel_upsample,
                # prenorm=prenorm,
                # dropout=dropres,
                # layer=layer,
                # residual=residual if residual is not None else 'R',
                # norm=norm,
                # pool=None,
            )


        # UNet backbone
        H = d_model
        # Down blocks
        d_layers = []
        for p in pool:
            if self.unet:
                for i in range(n_layers):
                    d_layers.append(_residual(H, L))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, H*expand, pool=p))
            L //= p
            H *= expand
        self.d_layers = nn.ModuleList(d_layers)

        # Center block
        c_layers = [ ]
        for i in range(n_layers):
            c_layers.append(_residual(H, L))
        self.c_layers = nn.ModuleList(c_layers)

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            H //= expand
            L *= p
            u_layers.append(UpPool(H*expand, H, pool=p)) # TODO

            for i in range(n_layers):
                u_layers.append(_residual(H, L))
        self.u_layers = nn.ModuleList(u_layers)

        self.norm = TransposedLN(d_model)

        # final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(d_model, d_model, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(d_model, out_channels))

    def forward(self, input_data, mel_spec=None):
        audio, diffusion_steps = input_data

        x = audio
        x = self.init_conv(x)

        # x = self.residual_layer((x, diffusion_steps))
        # x, diffusion_steps = input_data

        # embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # pass all UNet layers
        # Down blocks
        outputs = [] # Store all layers for SequenceUNet structure
        for layer in self.d_layers:
            outputs.append(x)
            x = layer(x, diffusion_step_embed, mel_spec=mel_spec)

        # Center block
        outputs.append(x)
        for layer in self.c_layers:
            x = layer(x, diffusion_step_embed, mel_spec=mel_spec)
        x = x + outputs.pop()

        for layer in self.u_layers:
            x = layer(x, diffusion_step_embed, mel_spec=mel_spec)
            if isinstance(layer, UpPool) or self.unet:
                x = x + outputs.pop()

        # Output embedding
        x = self.norm(x)
        x = self.final_conv(x)

        return x

    def __repr__(self):
        return f"sashimi_h{self.d_model}_d{self.n_layers}_pool{''.join(self.pool)}_expand{self.expand}_ff{self.ff}_{'uncond' if self.unconditional else 'cond'}"

    @classmethod
    def name(cls, cfg):
        return "{}_d{}_n{}_pool_{}_expand{}_ff{}".format(
            "unet" if cfg["unet"] else "snet",
            cfg["d_model"],
            cfg["n_layers"],
            len(cfg["pool"]),
            cfg["expand"],
            cfg["ff"],
        )

if __name__ == '__main__':
    model = Sashimi(n_layers=2).cuda()
    # Print parameter count
    print(sum(p.numel() for p in model.parameters()))

    model.eval()

    with torch.no_grad():
        # Forward in convolutional mode: used for training SaShiMi
        x = torch.randn(3, 1, 10240).cuda()
        steps = torch.randint(0, 4, (3, 1)).cuda()
        y = model((x, steps))
        print(y.shape)
