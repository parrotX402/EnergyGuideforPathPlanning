# near_infer.py
import os
import re
import numpy as np
import torch
import torch.nn as nn


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _linear_indices(sd, prefix):
    """
    找到类似 'encoder.encoder.0.weight' 这样的所有线性层序号（0,2,4,...）
    prefix: 'encoder.encoder.' or 'decoder.0.encoder.'
    """
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.weight$")
    idx = []
    for k in sd.keys():
        m = pat.match(k)
        if m:
            idx.append(int(m.group(1)))
    idx = sorted(idx)
    if len(idx) == 0:
        raise KeyError(f"Cannot find any Linear weights with prefix='{prefix}'. "
                       f"Available keys example: {list(sd.keys())[:5]}")
    return idx


def infer_mlp_dims_from_state_dict(sd):
    # encoder.encoder.*.weight
    enc_idx = _linear_indices(sd, "encoder.encoder.")
    # decoder.0.encoder.*.weight
    dec_idx = _linear_indices(sd, "decoder.0.encoder.")

    def wshape(name):
        return tuple(sd[name].shape)  # (out, in)

    # encoder dims: in -> h1 -> h2 -> ... -> latent
    enc_dims = []
    for i, layer_id in enumerate(enc_idx):
        out_f, in_f = wshape(f"encoder.encoder.{layer_id}.weight")
        if i == 0:
            enc_dims.append(in_f)
        enc_dims.append(out_f)

    # decoder dims: latent -> ... -> out
    dec_dims = []
    for i, layer_id in enumerate(dec_idx):
        out_f, in_f = wshape(f"decoder.0.encoder.{layer_id}.weight")
        if i == 0:
            dec_dims.append(in_f)
        dec_dims.append(out_f)

    return enc_dims, dec_dims


class MLP(nn.Module):
    """Linear + ELU 堆叠（最后一层不加激活）"""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """为了对齐 key: decoder.0.encoder.*"""
    def __init__(self, encoder_net: nn.Sequential):
        super().__init__()
        self.encoder = encoder_net


class NearLite(nn.Module):
    """
    复现 ckpt 里出现的结构：
      encoder.encoder (MLP)
      decoder.0.encoder (MLP)
      decoder.1 = ELU
    """
    def __init__(self, sd_no_module):
        super().__init__()

        enc_dims, dec_dims = infer_mlp_dims_from_state_dict(sd_no_module)

        # encoder.encoder.*
        self.encoder = nn.Module()
        self.encoder.encoder = MLP(enc_dims).net

        # decoder.0.encoder.* + decoder.1
        self.decoder = nn.Sequential(
            DecoderBlock(MLP(dec_dims).net),
            nn.ELU()
        )

    def forward(self, x56):
        z = self.encoder.encoder(x56)
        e = self.decoder[0].encoder(z)
        e = self.decoder[1](e)
        return e  # [B,1]


class NearEnergyModel:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(os.path.expanduser(ckpt_path), map_location=self.device)
        sd = ckpt[0] if isinstance(ckpt, list) else ckpt
        if not isinstance(sd, dict):
            raise TypeError(f"checkpoint must contain a state_dict(dict). got {type(sd)}")
        sd = strip_module_prefix(sd)

        # sanity check
        required = ["encoder.encoder.0.weight", "decoder.0.encoder.0.weight"]
        for k in required:
            if k not in sd:
                raise KeyError(f"Missing key '{k}' in ckpt. "
                               f"Maybe wrong checkpoint? First keys: {list(sd.keys())[:10]}")

        self.net = NearLite(sd).to(self.device)
        self.net.load_state_dict(sd, strict=True)
        self.net.eval()

    @torch.no_grad()
    def energy(self, x56: np.ndarray) -> float:
        x = torch.tensor(x56, dtype=torch.float32, device=self.device).unsqueeze(0)
        e = self.net(x)
        return float(e.item())

    @torch.no_grad()
    def shaped_reward(self, x56: np.ndarray, scale: float = 3.0,
                  clip_min: float = -0.2, clip_max: float = 0.0) -> float:
        E = self.energy(x56)

        # ✅ 只惩罚偏离：E=-1 -> 0, E> -1 -> negative
        r = (-(E) - 1.0) / float(scale)

        return float(np.clip(r, clip_min, clip_max))

