"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
This file is adapted from 
https://github.com/d-li14/efficientnetv2.pytorch
Original license: MIT
"""

import torch
import torch.nn as nn
import math
import torch.utils.checkpoint as checkpoint

# Define which models are available when importing *
__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    Ensure channel numbers are divisible by a certain number (default 8)
    Args:
        v: original channel number
        divisor: number to be divisible by
        min_value: minimum allowed channel number
    Returns:
        adjusted channel number divisible by divisor
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Prevent rounding down too much (not more than 10%)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For backward compatibility with older PyTorch
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer for channel-wise attention
    """
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        # Global average pooling to get channel descriptors
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                # Reduce dimension and apply SiLU
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                # Expand back to original channel number
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()  # Channel-wise gating
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # (B, C)
        y = self.fc(y).view(b, c, 1, 1, 1)  # reshape for broadcasting
        return x * y  # scale input channels


def conv_3x3_bn(inp, oup, stride):
    """
    3x3 3D convolution with BatchNorm and SiLU
    """
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    """
    1x1 3D convolution with BatchNorm and SiLU
    """
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv)
    Can optionally include Squeeze-and-Excitation (SE) module
    """
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        # Identity shortcut connection if stride=1 and input/output channels match
        self.identity = stride == 1 and inp == oup

        if use_se:
            # MBConv with SE module
            self.conv = nn.Sequential(
                # Pointwise convolution (expansion)
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                # Depthwise convolution
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                # Squeeze-and-Excitation
                SELayer(inp, hidden_dim),
                # Pointwise linear projection
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            # Fused MBConv (no SE)
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.identity:
            # Residual connection
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNet3D(nn.Module):
    """
    Full 3D EfficientNet model for classification
    """
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNet3D, self).__init__()
        self.cfgs = cfgs

        # First convolutional layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(1, input_channel, 2)]  # 1 input channel (e.g., grayscale CT)

        # Build MBConv blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # Final convolutional layer before pooling
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        # Initialize weights
        self._initialize_weights()

    def encoder(self, x):
        """
        Extract feature representation (without returning logits)
        """
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        logit = self.classifier(feature)
        return feature, logit

    def forward(self, x):
        """
        Forward pass returns both classification logits and feature vector
        """
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)
        return x, feature

    def _initialize_weights(self):
        """
        Initialize Conv3d, BatchNorm3d, and Linear layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


class EffNet3DEncoder(nn.Module):
    """
    Encoder variant of 3D EfficientNet (no classifier)
    """
    def __init__(self, cfgs, width_mult=1.):
        super(EffNet3DEncoder, self).__init__()
        self.cfgs = cfgs

        # Initial convolution
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(1, input_channel, 2)]

        # MBConv blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # Final conv layer and global pooling
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, feature_dim)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNet3D(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNet3D(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNet3D(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNet3D(cfgs, **kwargs)


def effnetv2_encoder_l(**kwargs):
    """
    Constructs a EfficientNetV2-L encoder (no classifier)
    """
    cfgs = [
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNet3DEncoder(cfgs, **kwargs)


if __name__ == "__main__":
    # Simple test for model instantiation and forward pass
    from time import sleep
    gpu_id = 1
    device = torch.device("cuda:{}".format(gpu_id))

    # Instantiate EfficientNetV2-L with 37 output classes
    network = effnetv2_l(num_classes=37).to(device)

    def count_parameters(model):
        """
        Count trainable parameters in the model
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    with torch.no_grad():
        # Test forward pass with dummy input (B=1, C=1, D=150, H=300, W=300)
        x = torch.zeros((1, 1, 150, 300, 300)).to(device)
        print(network(x)[0].shape)  # Logits shape

        sleep(5)
