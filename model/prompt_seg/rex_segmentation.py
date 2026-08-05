import torch
import torch.nn as nn
import torch.nn.functional as F

from model.efficient_net import EffNet3D


class RexGroundingCTSeg(nn.Module):
    """
    Text-guided 3D CT segmentation model based on ReXGroundingCT architecture.

    Args:
        text_dim: dimension of the text embedding / token features
            (must match the pre-computed embeddings stored on disk).
            E.g. 2560 for Qwen3 4B last hidden.
        llm_hidden_size: kept for backward compatibility / EffNet3D API
            (unused for the segmentation path).
        encoder_cfgs: the stage config for EffNet3D (7 stages, XL by default).
        num_classes_cls: classifier dim kept so pre-trained classifier weights
            load cleanly; classifier is not used for segmentation.
        vision_pretrained: optional path to pre-trained vision encoder weights.
    """

    def __init__(self,
                 text_dim: int = 2560,
                 llm_hidden_size: int = 1024,
                 encoder_cfgs=None,
                 num_classes_cls: int = 37,
                 vision_pretrained=None,):
        super().__init__()
        self.text_dim = text_dim
        self.llm_hidden_size = llm_hidden_size  # kept for interface compatibility

        from model.prompt_seg.seg_head import TextGuidedSegmentationHead

        # --- Image encoder ---
        cfgs = [
            # t, c, n, s, SE
            [1, 32, 4, 1, 0],
            [4, 64, 8, 2, 0],
            [4, 96, 8, 2, 0],
            [4, 192, 16, 2, 1],
            [6, 256, 24, 1, 1],
            [6, 512, 32, 2, 1],
            [6, 640, 8, 1, 1],
        ]

        self.visual_encoder = EffNet3D(cfgs, num_classes=num_classes_cls)
        if vision_pretrained:
            pt = torch.load(vision_pretrained, map_location='cpu')
            convert = False
            for name in list(pt.keys()):
                if "visual_transformer." in name:
                    convert = True
            if convert:
                print("Tag 'visual_transformer.' found in state dict - fixing!")
                for key in list(pt.keys()):
                    pt[key.replace("visual_transformer.", "")] = pt.pop(key)

            net_state_dict = self.visual_encoder.state_dict()
            for key in list(pt.keys()):
                new_key = key + "_pre"
                if key in list(net_state_dict.keys()) and pt[key].size() != net_state_dict[key].size():
                    pt[new_key] = pt.pop(key)
            msg = self.visual_encoder.load_state_dict(pt, strict=False)
            t = []
            for k in msg.unexpected_keys:
                if "text_transformer" not in k:
                    t.append(k)
            print("Unexpected keys in visual encoder:", t)
            print("Missing keys in visual encoder:", msg.missing_keys)

        # --- Decoder: infer encoder channels with a tiny dry run ---
        enc_ch = self._infer_encoder_channels(self.visual_encoder)
        assert len(enc_ch) == 4, f"expected 4 scales, got {len(enc_ch)}"
        self.seg_head = TextGuidedSegmentationHead(
            encoder_channels=tuple(enc_ch),
            text_dim=self.text_dim,
            decoder_channels=(256, 128, 64),
        )

        # --- Loss ---
        self.bce_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def _infer_encoder_channels(encoder: EffNet3D) -> list:
        """
        Probe the channel counts produced by `extract_multi_scale_features` using a
        tiny dummy volume.  This is a lazy way to ensure the decoder always matches
        the encoder, regardless of which EffNet cfg variant is used.
        """
        encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 64, 64)
            feats = encoder.extract_multi_scale_features(dummy)
        encoder.train()
        return [f.shape[1] for f in feats]

    @staticmethod
    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0):
        """
        Dice loss for binary segmentation.

        Args:
            logits: raw predictions (before sigmoid), shape (B, 1, D, H, W)
            targets: ground truth binary masks, shape (B, 1, D, H, W)
            smooth: smoothing constant to avoid division by zero
        Returns:
            scalar dice loss (1 - mean_dice)
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersect = (probs * targets).sum(dim=(2, 3, 4))
        denom = (probs + targets).sum(dim=(2, 3, 4)) + smooth
        dice = (2.0 * intersect + smooth) / denom
        return (1 - dice).mean()

    def forward(self, image, mask_target=None, embed_dict=None):
        """
        Args:
            image: (B, 1, D, H, W) CT volume (normalized HU).
            mask_target: optional (B, 1, D, H, W) — binary ground-truth mask.
            embed_dict: REQUIRED pre-computed text embeddings dict with keys
                "text_emb"  — (B, text_dim) pooled vector
                "text_tokens" — (B, L, text_dim) per-token features
                "text_mask"   — (B, L) attention mask (1 = valid, 0 = pad)

        Returns:
            logits: (B, 1, D, H, W) raw logits (before sigmoid).
            loss_dict: {"bce", "dice", "total"} if mask_target is given, else {}.
        """
        if embed_dict is None:
            raise ValueError(
                "RexGroundingCTSeg requires pre-computed text embeddings via `embed_dict`. "
                "Text encoder is no longer bundled with the segmentation module."
            )
        multi_scale = self.visual_encoder.extract_multi_scale_features(image)
        text_emb, text_tokens, text_mask = (
            embed_dict["text_emb"],
            embed_dict["text_tokens"],
            embed_dict["text_mask"],
        )

        logits = self.seg_head(multi_scale, text_emb, text_tokens, text_mask)

        # The decoder upsamples to `stem_spatial * 2` which is the input
        # resolution.  Ensure exact match in case of odd-sized inputs.
        if logits.shape[2:] != image.shape[2:]:
            logits = F.interpolate(logits, size=image.shape[2:],
                                    mode="trilinear", align_corners=False)

        loss_dict = {}
        if mask_target is not None:
            tgt = mask_target.float()
            bce = self.bce_loss(logits, tgt)
            dice = self.dice_loss(logits, tgt)
            total = 0.5 * bce + 0.5 * dice
            loss_dict = {"bce": bce, "dice": dice, "total": total}

        return logits, loss_dict

    @torch.no_grad()
    def predict(self, image, embed_dict=None):
        """Inference helper: returns probabilities in [0, 1]."""
        if embed_dict is None:
            raise ValueError(
                "RexGroundingCTSeg.predict requires pre-computed text embeddings "
                "via `embed_dict` (text encoder removed from the module)."
            )
        logits, _ = self.forward(image, mask_target=None, embed_dict=embed_dict)
        return torch.sigmoid(logits)
