import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3_bn(in_c, out_c, stride=1):
    """3x3 3D convolution with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


def conv1x1_bn(in_c, out_c):
    """1x1 3D convolution with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 1, 1, 0, bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


class TextGuidedFiLM(nn.Module):
    """
    FiLM-style conditioning: feature = gamma * feature + beta
    gamma, beta predicted from text embedding.
    """

    def __init__(self, text_dim: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.beta = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor):
        """
        Args:
            x: (B, C, D, H, W) feature map
            text_emb: (B, text_dim) pooled text embedding
        Returns:
            conditioned feature map of same shape as x
        """
        B, C, _, _, _ = x.shape
        gamma = self.gamma(text_emb).view(B, C, 1, 1, 1)
        beta = self.beta(text_emb).view(B, C, 1, 1, 1)
        return x * gamma + beta


class TextCrossAttention3D(nn.Module):
    """
    Cross-attention: image spatial tokens attend over text tokens.

    x: (B, C, D, H, W)
    text_tokens: (B, L, D_text)
    text_mask: (B, L)  (1 = valid, 0 = pad)
    """

    def __init__(self, feature_dim: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.num_heads = num_heads

        self.k_proj = nn.Linear(text_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(text_dim, feature_dim, bias=False)
        self.q_proj = nn.Conv3d(feature_dim, feature_dim, 1, bias=False)
        self.out = nn.Conv3d(feature_dim, feature_dim, 1, bias=False)
        self.norm = nn.BatchNorm3d(feature_dim)

    def forward(self, x, text_tokens, text_mask=None):
        """
        Args:
            x: (B, C, D, H, W) spatial feature map
            text_tokens: (B, L, D_text) per-token text features
            text_mask: (B, L) optional attention mask
        Returns:
            attended feature map with residual connection
        """
        B, C, D, H, W = x.shape
        q = self.q_proj(x).view(B, self.num_heads, C // self.num_heads, D * H * W)
        k = self.k_proj(text_tokens).view(B, text_tokens.shape[1], self.num_heads, C // self.num_heads)
        v = self.v_proj(text_tokens).view(B, text_tokens.shape[1], self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        scale = 1.0 / (float(C // self.num_heads) ** 0.5)
        attn = torch.matmul(q.permute(0, 1, 3, 2).contiguous(), k.transpose(-1, -2)) * scale
        if text_mask is not None:
            mask = text_mask.unsqueeze(1).unsqueeze(1).float()
            attn = attn.masked_fill(mask < 0.5, -1e9)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, D, H, W)
        out = self.out(out)
        return self.norm(out + x)


class DecoderBlock(nn.Module):
    """
    Takes a deep feature and a shallower skip feature, upsamples deep,
    fuses, applies text-conditioned FiLM and cross-attention.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, text_dim: int):
        super().__init__()
        self.fuse = conv3x3_bn(in_ch + skip_ch, out_ch)
        self.conv = conv3x3_bn(out_ch, out_ch)
        self.film = TextGuidedFiLM(text_dim, out_ch)
        self.attn = TextCrossAttention3D(out_ch, text_dim, num_heads=4)

    def forward(self, x, skip, text_emb, text_tokens, text_mask):
        """
        Args:
            x: deep feature (B, in_ch, Dd, Hd, Wd)
            skip: skip feature (B, skip_ch, Ds, Hs, Ws)
            text_emb: (B, text_dim) global text embedding
            text_tokens: (B, L, text_dim) per-token text features
            text_mask: (B, L) optional text mask
        Returns:
            fused and upsampled feature (B, out_ch, Ds, Hs, Ws)
        """
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.conv(x)
        x = self.film(x, text_emb)
        x = self.attn(x, text_tokens, text_mask)
        return x


class TextGuidedSegmentationHead(nn.Module):
    """
    Text-guided 3D segmentation decoder.

    Args:
        encoder_channels: list of 4 channel counts (from shallow to deep).
                          E.g. [24, 256, 512, 1792] for default EffNet XL.
        text_dim: dimension of the text embedding (Qwen3 hidden size).
        decoder_channels: intermediate channel counts. 3 ints for the
                          3 decoder blocks (deep -> shallow).
    """

    def __init__(self, encoder_channels=(24, 256, 512, 1792),
                 text_dim: int = 1024,
                 decoder_channels=(256, 128, 64)):
        super().__init__()
        assert len(encoder_channels) == 4, "need 4 scales"
        assert len(decoder_channels) == 3, "need 3 decoder channels"

        self.text_dim = text_dim

        proj_ch = [decoder_channels[2], decoder_channels[1], decoder_channels[0], decoder_channels[0]]
        self.input_projs = nn.ModuleList([
            conv1x1_bn(encoder_channels[i], proj_ch[i]) for i in range(4)
        ])
        self._proj_ch = proj_ch

        self.text_film_deep = TextGuidedFiLM(text_dim, proj_ch[-1])
        self.text_attn_deep = TextCrossAttention3D(proj_ch[-1], text_dim, num_heads=4)

        self.decoders = nn.ModuleList([
            DecoderBlock(decoder_channels[0], proj_ch[2], decoder_channels[0], text_dim),
            DecoderBlock(decoder_channels[0], proj_ch[1], decoder_channels[1], text_dim),
            DecoderBlock(decoder_channels[1], proj_ch[0], decoder_channels[2], text_dim),
        ])

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose3d(decoder_channels[-1], decoder_channels[-1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            conv3x3_bn(decoder_channels[-1], 32),
            nn.Conv3d(32, 1, kernel_size=1),
        )

    def forward(self, multi_scale_features, text_emb, text_tokens, text_mask=None):
        """
        Args:
            multi_scale_features: list of 4 tensors, shallowest first.
            text_emb: (B, text_dim) global pooled text feature
            text_tokens: (B, L, text_dim)
            text_mask: (B, L) optional; if None assumes all tokens valid.
        Returns:
            logits: (B, 1, D, H, W) raw segmentation logits
        """
        projected = [self.input_projs[i](multi_scale_features[i]) for i in range(4)]

        x = projected[-1]
        x = self.text_film_deep(x, text_emb)
        x = self.text_attn_deep(x, text_tokens, text_mask)

        skips = [projected[2], projected[1], projected[0]]
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip, text_emb, text_tokens, text_mask)

        x = self.final_upsample(x)

        logits = self.final(x)
        return logits
