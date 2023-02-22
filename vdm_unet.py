import numpy as np
import torch
from torch import einsum, nn, pi, softmax

from utils import zero_init


class UNetVDM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        attention_params = dict(
            n_heads=cfg.n_attention_heads,
            n_channels=cfg.embedding_dim,
            norm_groups=cfg.norm_groups,
        )
        resnet_params = dict(
            ch_in=cfg.embedding_dim,
            ch_out=cfg.embedding_dim,
            condition_dim=4 * cfg.embedding_dim,
            dropout_prob=cfg.dropout_prob,
            norm_groups=cfg.norm_groups,
        )
        if cfg.use_fourier_features:
            self.fourier_features = FourierFeatures()
        self.embed_conditioning = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.embedding_dim * 4, cfg.embedding_dim * 4),
            nn.SiLU(),
        )
        total_input_ch = cfg.input_channels
        if cfg.use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features
        self.conv_in = nn.Conv2d(total_input_ch, cfg.embedding_dim, 3, padding=1)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if cfg.attention_everywhere
                else None,
            )
            for _ in range(cfg.n_blocks)
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        self.mid_attn_block = AttentionBlock(**attention_params)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        # Up path: n_blocks+1 blocks with a resnet block and maybe attention.
        resnet_params["ch_in"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if cfg.attention_everywhere
                else None,
            )
            for _ in range(cfg.n_blocks + 1)
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=cfg.norm_groups, num_channels=cfg.embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(cfg.embedding_dim, cfg.input_channels, 3, padding=1)),
        )

    def forward(self, z, g_t):
        # Get gamma to shape (B, ).
        g_t = g_t.expand(z.shape[0])  # assume shape () or (1,) or (B,)
        assert g_t.shape == (z.shape[0],)
        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        t = (g_t - self.cfg.gamma_min) / (self.cfg.gamma_max - self.cfg.gamma_min)
        t_embedding = get_timestep_embedding(t, self.cfg.embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding)

        h = self.maybe_concat_fourier(z)
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            hs.append(h)
            h = down_block(h, cond)
        hs.append(h)
        h = self.mid_resnet_block_1(h, cond)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, cond)
        for up_block in self.up_blocks:  # n_blocks+1 times
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, cond)
        prediction = self.conv_out(h)
        assert prediction.shape == z.shape, (prediction.shape, z.shape)
        return prediction + z

    def maybe_concat_fourier(self, z):
        if self.cfg.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        return z


class ResnetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out=None,
        condition_dim=None,
        dropout_prob=0.0,
        norm_groups=32,
    ):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_out = ch_out
        self.condition_dim = condition_dim
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
        )
        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        h = self.net1(x)
        if condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            condition = self.cond_proj(condition)
            condition = condition[:, :, None, None]
            h = h + condition
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims)


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x
