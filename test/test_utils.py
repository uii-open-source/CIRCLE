import os
import math
import numpy as np
import pandas as pd
import json
import SimpleITK as sitk
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from model.circle import CIRCLE


def load_model(vision_encoder_dir, text_encoder_dir, num_classes):
    tokenizer = BertTokenizer.from_pretrained(text_encoder_dir)
    text_encoder = BertModel.from_pretrained(text_encoder_dir)
    
    circle_model = CIRCLE(
    text_encoder=text_encoder,
    dim_image=1792,
    dim_text=768,
    dim_latent=512,
    num_classes=num_classes,
    )
    circle_model.load(vision_encoder_dir)
    return circle_model, tokenizer


def crop_image(image, center, spacing, crop_size, axes, default_value=-1024):
    direction = []
    for col in range(3):
        for row in range(3):
            direction.append(float(axes[col][row]))

    offset = [0.0, 0.0, 0.0]
    for dim in range(3):
        half_len = (crop_size[dim] - 1) * spacing[dim] / 2.0
        for axis in range(3):
            offset[axis] += half_len * axes[dim][axis]
    origin = [float(center[i] - offset[i]) for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize([int(s) for s in crop_size])
    resampler.SetOutputSpacing([float(s) for s in spacing])
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())

    output_image = resampler.Execute(image)
    return output_image


def intensity_normalize(img_data, mean, stddev, clip=False):
    img_data = ((img_data + mean) / stddev).astype(np.float32)
    if clip:
        img_data = np.clip(img_data, -1.0, 1.0)
    return img_data


def prepare_image(img_path, crop_center):

        mean = 0
        stddev = 1000
        clip = False

        crop_size = np.array([300, 300, 160])
        crop_spacing = np.array([1.0, 1.0, 2.0])

        crop_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)
        crop_scale_ratio = np.array([1.0, 1.0, 1.0])

        img = sitk.ReadImage(img_path)
        cropped_img = crop_image(img, crop_center, crop_spacing, crop_size, crop_axes)
        cropped_img = sitk.GetArrayFromImage(cropped_img)
        cropped_img = intensity_normalize(cropped_img, mean, stddev, clip)

        cropped_img = torch.from_numpy(cropped_img)
        cropped_img = torch.unsqueeze(cropped_img, 0)
        cropped_img = cropped_img.float()
        return cropped_img


def get_lung_center(center_csv):
        df = pd.read_csv(center_csv)
        image_to_center = {}
        for i in range(df.shape[0]):
            image_to_center[df.loc[i, 'image_name']] = np.array([df.loc[i, 'lung_center_world_x'],
                                                                 df.loc[i, 'lung_center_world_y'],
                                                                 df.loc[i, 'lung_center_world_z']])
        return image_to_center


def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=1)
    softmax_array = softmax(array)
    return softmax_array


def model_infer(
    vision_encoder_dir,
    text_encoder_dir,
    image_path,
    crop_center,
    text_list,
    device,
    num_classes=37
):
    circle_model, tokenizer = load_model(vision_encoder_dir, text_encoder_dir, num_classes)
    circle_model = circle_model.to(device)
    cropped_image = prepare_image(image_path, crop_center)
    
    clip_prob_list = []
    with torch.no_grad():
        image_tensor_gpu = torch.unsqueeze(cropped_image, dim=0).to(device)
        visual_feature, cls_logits = circle_model.encode_image_and_cls(image_tensor_gpu)
        
        text_features = []
        for text in text_list:
            text_tokens = tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            text_feature = circle_model.encode_text(text_tokens)
            text_features.append(text_feature)
        text_features = torch.stack(text_features, dim=0)

        for i in range(len(text_list)):
            output_ori = circle_model.clip_forward(visual_feature, text_features[i])
            output = apply_softmax(output_ori)
            append_out = output.detach().cpu().numpy()
            clip_prob_list.append(append_out[0, 0])
    
    return clip_prob_list, cls_logits, visual_feature.detach().cpu()[0].numpy()


# if __name__ == '__main__':
#     vision_encoder_dir = 
#     text_encoder_dir,
#     image_path,
#     crop_center,
#     text_list,
#     device