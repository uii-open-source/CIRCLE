import numpy as np
import torch
import torch.nn.functional as F

from collections import OrderedDict
from einops import rearrange
from pathlib import Path
from torch import nn, einsum

from model.efficient_net import EffNet3D


def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)


def load_model_ckpt(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = val

    msg = model.load_state_dict(new_state_dict, strict=True)
    print('load effnet encoder: {}'.format(msg))

    return model


class CIRCLE(nn.Module):
    def __init__(
        self,
        text_encoder,
        dim_text = 512,
        dim_image = 1792,
        dim_latent = 512,
        **kwargs
    ):
        super().__init__()
       
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.text_transformer = text_encoder
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
        self.visual_transformer = EffNet3D(cfgs, num_classes=37)
        self.to_image_latent = nn.Linear(dim_image, dim_latent, bias=False)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(10))
        self.bias = nn.Parameter(torch.ones([]) * (-10))

        self.cls_loss_fn = nn.BCEWithLogitsLoss()

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path), map_location="cpu")
        msg = self.load_state_dict(pt, strict=False)
        print(msg)

    def encode_text(self, text):
        text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask)
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:,0,:].contiguous()
        text_latents = self.to_text_latent(text_embeds)
        return text_latents

    def encode_image(self, image):
        cls_logits, enc_image = self.visual_transformer(image)
        image_latents = self.to_image_latent(enc_image)
        return image_latents

    def run_classification(self, image):
        cls_logits, enc_image = self.visual_transformer(image)
        return cls_logits, enc_image

    def clip_forward(self, image_features, text_features):
        image_features = image_features[:,0,:].contiguous()
        image_features = l2norm(image_features)
        text_features = l2norm(text_features)
        output = image_features @ text_features.T * self.temperature.exp() + self.bias
        return output

    def forward(
        self,
        text,
        image,
        cls_labels,
        device,
    ):
        cls_logits, enc_image = self.visual_transformer(image)  # (b, c, d, h, w)
        image_embeds = enc_image
        image_latents = self.to_image_latent(image_embeds)
        
        text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
        text_embeds = text_embeddings[0]
        text_embeds = text_embeds[:,0,:].contiguous()
        text_latents = self.to_text_latent(text_embeds)
        
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))
        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = 1).contiguous()
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = 1).contiguous()

        temp = self.temperature.exp()
        text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
        image_to_text = rearrange(text_to_image, '... t i -> ... i t')

        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # SigLIP loss
        logits = torch.squeeze(image_to_text, 0) + self.bias
        labels = -torch.ones((logits.shape[0], logits.shape[0]), device=device, dtype=logits.dtype)
        labels = 2 * torch.eye(logits.shape[0], device=device, dtype=logits.dtype) + labels
        clip_loss = -F.logsigmoid(labels * logits).sum() / logits.shape[0]

        # cls loss
        cls_loss = self.cls_loss_fn(cls_logits, cls_labels)

        loss = 0.9 * cls_loss + 0.1 * clip_loss

        return loss, cls_loss, clip_loss

