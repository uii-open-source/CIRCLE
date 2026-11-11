import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    """
    Ensure channel number is divisible by `divisor`.
    Commonly used in EfficientNet-style width scaling.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    class SiLU(nn.Module):
        """
        Fallback SiLU activation for older PyTorch versions.
        """
        def forward(self, x):
            return x * torch.sigmoid(x)

class SELayer(nn.Module):
    """
    3D squeeze-and-excitation block for channel re-weighting.
    """

    def __init__(self, inp, oup, reduction=4):
        """
        Args:
            inp: input channel count (used for reduced hidden size)
            oup: output channel count
            reduction: squeeze ratio
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(inplace=True),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Re-calibrate channel responses using global context.
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

def conv_3x3_bn(inp, oup, stride):
    """
    3x3 Conv3D + BatchNorm3D + SiLU block.
    """
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )

def conv_1x1_bn(inp, oup):
    """
    1x1 Conv3D + BatchNorm3D + SiLU block.
    """
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )

class MBConv(nn.Module):
    """
    Mobile inverted bottleneck block (3D variant), optionally with SE.
    """

    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        """
        Args:
            inp: input channels
            oup: output channels
            stride: spatial stride (1 or 2)
            expand_ratio: expansion factor for hidden channels
            use_se: whether to include squeeze-excitation
        """
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        # Residual connection only when shape is unchanged
        self.identity = stride == 1 and inp == oup
        if use_se:
            # Expand -> depthwise conv -> SE -> project
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(inplace=True),
                SELayer(inp, hidden_dim),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            # Lightweight alternative without SE
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(inplace=True),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        """
        Forward MBConv with optional residual shortcut.
        """
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EffNet3DEncoderForSeg(nn.Module):
    """
    3D EfficientNet-style encoder for segmentation.
    Returns multi-scale features for UNet-like decoder skip connections.
    """

    def __init__(self, pretrained: str = None):
        """
        Args:
            pretrained: reserved argument for compatibility
        """
        super(EffNet3DEncoderForSeg, self).__init__()
        # (expand_ratio, channels, repeats, stride, use_se)
        cfgs = [
            [1, 32, 4, 1, 0],
            [4, 64, 8, 2, 0],
            [4, 96, 8, 2, 0],
            [4, 192, 16, 2, 1],
            [6, 256, 24, 1, 1],
            [6, 512, 32, 2, 1],
            [6, 640, 8, 1, 1],
        ]
       
        # Initial stem downsampling (/2)
        self.stem = conv_3x3_bn(1, 24, 2)
       
        # Encoder stages
        self.stage0 = self._make_stage(24, 32, cfgs[0])
        self.stage1 = self._make_stage(32, 64, cfgs[1])
        self.stage2 = self._make_stage(64, 96, cfgs[2])
        self.stage3 = self._make_stage(96, 192, cfgs[3])
        self.stage4 = self._make_stage(192, 256, cfgs[4])
        self.stage5 = self._make_stage(256, 512, cfgs[5])
        self.stage6 = self._make_stage(512, 640, cfgs[6])
       
        # Final channel expansion for high-level representation
        self.final_conv = conv_1x1_bn(640, 1792)
       
        self._initialize_weights()

        # Load pretrained backbone mapping if available
        # if pretrained is not None:
        self.load_pretrained_weights()

    def _make_stage(self, in_ch, out_ch, config):
        """
        Construct one encoder stage from repeated MBConv blocks.
        """
        t, c, n, s, use_se = config
        layers = []
        for i in range(n):
            layers.append(MBConv(in_ch, out_ch, s if i == 0 else 1, t, use_se))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward encoder and return selected multi-scale features.
        Returns:
            list of feature tensors from shallow to deep scales
        """
        features = []
        x = self.stem(x)
       
        # Keep selected stage outputs for decoder skip connections
        x = self.stage0(x)
        features.append(x)
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        x = self.stage4(x)
        features.append(x)
        x = self.stage5(x)
        x = self.stage6(x)
       
        x = self.final_conv(x)
        features.append(x)
       
        return features

    def _initialize_weights(self):
        """
        Initialize module parameters with convolution/BN/linear defaults.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def load_pretrained_weights(self, device=None):
        """
        Load pretrained encoder weights by shape-based key matching.
        Args:
            device: target device for checkpoint loading
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_pt_path = ''
        
        print(f"Loading pretrained encoder weights from: {model_pt_path}")
        orig_state_dict = torch.load(model_pt_path, map_location=device)
        
        # Collect current model parameter keys
        current_model_dict = self.state_dict()
        encoder_keys_new = [k for k in current_model_dict.keys()]

        # Ignore classifier/projection heads from the source checkpoint
        backbone_keys_orig = [
            k for k in orig_state_dict.keys()
            if not k.startswith('classifier') and not k.startswith('visual_latent_layer')
        ]

        # Group original keys by tensor shape for robust matching
        orig_by_shape = {}
        for k in backbone_keys_orig:
            shape = tuple(orig_state_dict[k].shape)
            orig_by_shape.setdefault(shape, []).append(k)

        matched_keys = {}
        successful_mappings = []
        used_orig_keys = set()

        # Match each new key to best source key with same shape
        for new_key in encoder_keys_new:
            shape = tuple(current_model_dict[new_key].shape)
            candidates = orig_by_shape.get(shape, [])

            if not candidates:
                continue

            best_match = None
            best_score = -1

            # Heuristic scoring based on stage naming and layer semantics
            for cand in candidates:
                score = 0
                if 'stem' in new_key and 'stem' in cand:
                    score += 100
                if 'final_conv' in new_key and 'conv.' in cand:
                    score += 100
                if ('stage0' in new_key and cand.startswith('features.0')) or \
                   ('stage1' in new_key and cand.startswith('features.4')) or \
                   ('stage2' in new_key and cand.startswith('features.12')) or \
                   ('stage3' in new_key and cand.startswith('features.20')) or \
                   ('stage4' in new_key and cand.startswith('features.36')) or \
                   ('stage5' in new_key and cand.startswith('features.60')) or \
                   ('stage6' in new_key and cand.startswith('features.92')):
                    score += 50

                if ('conv' in new_key and 'conv' in cand) or ('bn' in new_key.lower() and 'batchnorm' in cand.lower()):
                    score += 10

                if score > best_score:
                    best_score = score
                    best_match = cand

            # Keep one-to-one mapping to avoid reusing source tensors
            if best_match and best_match not in used_orig_keys:
                matched_keys[new_key] = orig_state_dict[best_match]
                successful_mappings.append((best_match, new_key))
                used_orig_keys.add(best_match)
                orig_by_shape[shape].remove(best_match)
                if not orig_by_shape[shape]:
                    del orig_by_shape[shape]

        total_new = len(encoder_keys_new)
        matched = len(successful_mappings)
        unused_orig = len(backbone_keys_orig) - len(used_orig_keys)

        print(f"\nSuccessfully matched: {matched} / {total_new} parameters")
        print(f"Unmatched in new model: {total_new - matched}")
        print(f"Unused in original .pt: {unused_orig}")

        self.load_state_dict(matched_keys, strict=False)
        print(f"Pretrained encoder weights loaded successfully!")

def up_3x3_bn(inp, oup):
    """
    2x upsampling block via transposed Conv3D + BN + SiLU.
    """
    return nn.Sequential(
        nn.ConvTranspose3d(inp, oup, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )

class DecoderBlock(nn.Module):
    """
    UNet-style decoder block: upsample, fuse skip, refine with convs.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: input channels from deeper decoder level
            skip_channels: channels from encoder skip feature
            out_channels: output channels after fusion
        """
        super(DecoderBlock, self).__init__()
        self.upsample = up_3x3_bn(in_channels, out_channels)
        # Align skip channels to decoder channels when needed
        self.reduce_skip = nn.Conv3d(skip_channels, out_channels, 1) if skip_channels != out_channels else nn.Identity()
        self.conv1 = conv_3x3_bn(out_channels + out_channels, out_channels, 1)
        self.conv2 = conv_3x3_bn(out_channels, out_channels, 1)

    def forward(self, x, skip):
        """
        Forward decode step with skip fusion.
        """
        x = self.upsample(x)
        skip = self.reduce_skip(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationDecoder(nn.Module):
    """
    Multi-stage decoder that reconstructs dense segmentation logits.
    """

    def __init__(self, num_classes=3):
        """
        Args:
            num_classes: number of output segmentation classes
        """
        super(SegmentationDecoder, self).__init__()
        # Deep-to-shallow decoding path with encoder skips
        self.dec3 = DecoderBlock(1792, 256, 512)
        self.dec2 = DecoderBlock(512, 96, 256)
        self.dec1 = DecoderBlock(256, 64, 128)
        self.dec0 = DecoderBlock(128, 32, 64)
        self.stem_up = up_3x3_bn(64, 32)
        
        # Final 1x1 prediction head
        self.segmentation_head = nn.Conv3d(32, num_classes, kernel_size=1, bias=True)

    def forward(self, features):
        """
        Decode encoder feature pyramid to segmentation logits.
        Args:
            features: list of encoder features from `EffNet3DEncoderForSeg`
        Returns:
            logits tensor of shape (B, num_classes, D, H, W)
        """
        x = self.dec3(features[4], features[3])  
        x = self.dec2(x, features[2])           
        x = self.dec1(x, features[1])           
        x = self.dec0(x, features[0])           
        
        x = self.stem_up(x)         
        logits = self.segmentation_head(x)      
        return logits

class EffUNet3D(nn.Module):
    """
    End-to-end 3D segmentation network: Efficient encoder + UNet decoder.
    """

    def __init__(self, num_classes=3):
        """
        Args:
            num_classes: number of segmentation classes
        """
        super(EffUNet3D, self).__init__()
        self.encoder = EffNet3DEncoderForSeg()
        self.decoder = SegmentationDecoder(num_classes=num_classes)

    def forward(self, x):
        """
        Forward full segmentation model.
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits

def effunet3d_xl(num_classes=3):
    """
    Factory function for EffUNet3D model.
    """
    return EffUNet3D(num_classes=num_classes)


if __name__ == "__main__":
    """
    Minimal smoke test for model forward output shape.
    """
    import torch

    input_data = torch.randn(1, 1, 128, 128, 128)

    model = effunet3d_xl(num_classes=36) 

    model.eval()

    with torch.no_grad():
        output = model(input_data)

    print("Output shape:", output.shape)
