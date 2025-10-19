# Load open source software package
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from pathlib import Path
from torch import nn, einsum

# Import Efficientnet as image encoder
from model.efficient_net import EffNet3D

# Small utility function to compute logarithm with numerical stability
def log(t, eps=1e-20):
    """
    Compute logarithm of input tensor t, adding epsilon to avoid log(0)
    Args:
        t: input tensor
        eps: small number to avoid log(0)
    Returns:
        log(t + eps)
    """
    return torch.log(t + eps)

# L2 normalization along the last dimension
def l2norm(t):
    """
    L2 normalize the input tensor along the last dimension
    Args:
        t: input tensor of shape (..., D)
    Returns:
        normalized tensor with same shape
    """
    return F.normalize(t, dim=-1)

# Function to load a model checkpoint, optionally removing "module." prefix
def load_model_ckpt(model, ckpt_path):
    """
    Load pretrained weights into a model, handling DataParallel "module." prefix.
    Args:
        model: PyTorch model
        ckpt_path: path to checkpoint
    Returns:
        model with loaded weights
    """
    # Load checkpoint to CPU
    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    
    # Remove 'module.' prefix if it exists (for models trained with DataParallel)
    for key, val in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = val
    
    # Load state dict into model strictly
    msg = model.load_state_dict(new_state_dict, strict=True)
    print('load effnet encoder: {}'.format(msg))
    return model


class CIRCLE(nn.Module):
    """
    Main CIRCLE model combining image and text encoders for multi-modal learning.
    Implements image-text contrastive learning (CLIP-like) with classification loss.
    """
    def __init__(
        self,
        text_encoder,
        dim_text=512,
        dim_image=1792,
        dim_latent=512,
        num_classes=37,
        **kwargs
    ):
        """
        Initialize CIRCLE model.
        Args:
            text_encoder: transformer model for text
            dim_text: dimensionality of text embedding
            dim_image: dimensionality of image embedding from EffNet3D
            dim_latent: dimension of projected latent space for contrastive learning
        """
        super().__init__()
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        # Text transformer (e.g., BERT-like)
        self.text_transformer = text_encoder

        # Configuration for 3D EfficientNet (t, c, n, s, SE)
        cfgs = [
            [1, 32, 4, 1, 0],
            [4, 64, 8, 2, 0],
            [4, 96, 8, 2, 0],
            [4, 192, 16, 2, 1],
            [6, 256, 24, 1, 1],
            [6, 512, 32, 2, 1],
            [6, 640, 8, 1, 1],
        ]
        # Visual encoder
        self.visual_transformer = EffNet3D(cfgs, num_classes=num_classes)

        # Linear projection to latent space for image features
        self.to_image_latent = nn.Linear(dim_image, dim_latent, bias=False)
        # Linear projection to latent space for text features
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        # Learnable temperature and bias for contrastive scaling
        self.temperature = nn.Parameter(torch.ones(1) * np.log(10))
        self.bias = nn.Parameter(torch.ones(1) * (-10))

        # Binary cross-entropy loss for classification
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

    # Overriding state_dict/load_state_dict to preserve default behavior
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)
    
    # Load checkpoint from path
    def load(self, path):
        """
        Load model weights from file
        """
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path), map_location="cpu")
        msg = self.load_state_dict(pt, strict=False)
        print(msg)

    # Encode text to latent representation
    def encode_text(self, text):
        """
        Args:
            text: dict with input_ids and attention_mask
        Returns:
            text_latents: projected text embeddings
        """
        # Forward through text transformer
        text_embeddings = self.text_transformer(
            text.input_ids,
            attention_mask=text.attention_mask
        )
        # Take the [CLS] token embedding
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:, 0, :].contiguous()
        # Project to latent space
        text_latents = self.to_text_latent(text_embeds)
        return text_latents

    # Encode image to latent representation
    def encode_image(self, image):
        """
        Args:
            image: input 3D image tensor (B, C, D, H, W)
        Returns:
            image_latents: projected image embeddings
        """
        cls_logits, enc_image = self.visual_transformer(image)
        image_latents = self.to_image_latent(enc_image)
        return image_latents

    # Run classification head only
    def run_classification(self, image):
        """
        Args:
            image: input 3D image tensor
        Returns:
            cls_logits: classification logits
            enc_image: feature map from EffNet3D
        """
        cls_logits, enc_image = self.visual_transformer(image)
        return cls_logits, enc_image

    # Forward pass for CLIP-style contrastive similarity
    def clip_forward(self, image_features, text_features):
        """
        Args:
            image_features: (B, D) image features
            text_features: (B, D) text features
        Returns:
            scaled similarity matrix (B, B)
        """
        # L2 normalize features
        image_features = image_features[:, 0, :].contiguous()
        image_features = l2norm(image_features)
        text_features = l2norm(text_features)
        # Compute cosine similarity scaled by temperature and bias
        output = image_features @ text_features.T * self.temperature.exp() + self.bias
        return output

    # Full forward pass for training
    def forward(
        self,
        text,
        image,
        cls_labels,
        device,
    ):
        """
        Args:
            text: dict of input_ids and attention_mask
            image: 3D image tensor (B, C, D, H, W)
            cls_labels: ground truth classification labels
            device: torch device
        Returns:
            loss: combined loss
            cls_loss: classification loss
            clip_loss: contrastive loss
        """
        # Visual forward pass
        cls_logits, enc_image = self.visual_transformer(image)  # (B, C, D, H, W)
        image_embeds = enc_image
        # Project image embeddings to latent space
        image_latents = self.to_image_latent(image_embeds)

        # Text forward pass
        text_embeddings = self.text_transformer(
            text.input_ids,
            attention_mask=text.attention_mask
        )
        text_embeds = text_embeddings[0][:, 0, :].contiguous()
        # Project text embeddings to latent space
        text_latents = self.to_text_latent(text_embeds)

        # Normalize embeddings
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # Reshape to add 'm' dimension for multiple positives (currently m=1)
        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m=1).contiguous()
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m=1).contiguous()

        temp = self.temperature.exp()

        # Compute pairwise similarity between text and image latents
        text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
        image_to_text = rearrange(text_to_image, '... t i -> ... i t')

        # Flatten for loss computation
        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # SigLIP contrastive loss
        logits = torch.squeeze(image_to_text, 0) + self.bias
        # Prepare target labels: positive pairs = 1, negative pairs = -1
        labels = -torch.ones((logits.shape[0], logits.shape[0]), device=device, dtype=logits.dtype)
        labels = 2 * torch.eye(logits.shape[0], device=device, dtype=logits.dtype) + labels
        clip_loss = -F.logsigmoid(labels * logits).sum() / logits.shape[0]

        # Classification loss
        cls_loss = self.cls_loss_fn(cls_logits, cls_labels)

        # Weighted sum of losses
        loss = 0.9 * cls_loss + 0.1 * clip_loss

        return loss, cls_loss, clip_loss


