import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from model.circle import CIRCLE


def load_model(vision_encoder_dir, text_encoder_dir, num_classes=4):
    """
    Load the vision and text encoders from specified directories.

    Args:
        vision_encoder_dir (str): Directory path for vision encoder.
        text_encoder_dir (str): Directory path for text encoder.
        num_classes (int): Number of classes for classification.

    Returns:
        tuple: A tuple containing the CIRCLE model and BERT tokenizer.
    """
    # Initialize BERT tokenizer and model for text encoding
    tokenizer = BertTokenizer.from_pretrained(text_encoder_dir)
    text_encoder = BertModel.from_pretrained(text_encoder_dir)

    # Initialize the CIRCLE model
    circle_model = CIRCLE(
        text_encoder=text_encoder,
        dim_image=1792,
        dim_text=768,
        dim_latent=512,
        num_classes=num_classes,
    )
    # Load pre-trained weights into the CIRCLE model
    circle_model.load(vision_encoder_dir)
    return circle_model, tokenizer


def crop_image(image, center, spacing, crop_size, axes, default_value=-1024):
    """
    Crop a 3D image (SimpleITK Image) centered at a given point with specified size, spacing, and axes.
    Args:
        image: SimpleITK.Image, input 3D medical image
        center: list/array of length 3, center of crop in world coordinates
        spacing: list/array of length 3, desired voxel spacing of output crop
        crop_size: list/array of length 3, size of output crop in voxels
        axes: 3x3 rotation matrix, orientation of output crop
        default_value: value to fill outside original image (e.g., -1024 for CT)
    Returns:
        output_image: cropped and resampled SimpleITK.Image
    """
    direction = []
    # Flatten 3x3 axes matrix to 1D direction vector for SimpleITK
    for row in range(3):
        for col in range(3):
            direction.append(float(axes[col][row]))

    # Compute origin of the crop in world coordinates
    offset = [0.0, 0.0, 0.0]
    for dim in range(3):
        half_len = (crop_size[dim] - 1) * spacing[dim] / 2.0
        for axis in range(3):
            offset[axis] += half_len * axes[dim][axis]
    origin = [float(center[i] - offset[i]) for i in range(3)]

    # Set up SimpleITK resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize([int(s) for s in crop_size])
    resampler.SetOutputSpacing([float(s) for s in spacing])
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())  # identity transform

    # Execute resampling and return cropped image
    output_image = resampler.Execute(image)
    return output_image


def intensity_normalize(img_data, mean, stddev, clip=False):
    """
    Normalize image intensities to zero-mean and unit variance
    Args:
        img_data: numpy array, raw image data
        mean: mean value to shift image
        stddev: std deviation to scale image
        clip: whether to clip values to [-1, 1]
    Returns:
        normalized image as float32 numpy array
    """
    img_data = img_data.astype(np.float32)
    img_data = ((img_data - mean) / stddev).astype(np.float32)
    if clip:
        img_data = np.clip(img_data, -1.0, 1.0)
    return img_data


def prepare_image(img_path, crop_center):
    """
    Prepares an image for model input by cropping, normalizing intensity, and converting to tensor.

    Args:
        img_path (str): Path to the input image file.
        crop_center: Center coordinates for cropping the image.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    # Define normalization parameters and cropping specifications
    mean = -400
    stddev = 600
    clip = False

    crop_size = np.array([300, 300, 160])
    crop_spacing = np.array([1.0, 1.0, 2.0])
    crop_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)

    # Read image, crop according to specifications, and normalize intensity values
    img = sitk.ReadImage(img_path)
    cropped_img = crop_image(img, crop_center, crop_spacing, crop_size, crop_axes)
    cropped_img = sitk.GetArrayFromImage(cropped_img)
    cropped_img = intensity_normalize(cropped_img, mean, stddev, clip)

    # Convert the processed image array to a PyTorch tensor with appropriate format
    cropped_img = torch.from_numpy(cropped_img)
    cropped_img = torch.unsqueeze(cropped_img, 0)
    cropped_img = cropped_img.float()
    return cropped_img


def get_lung_center(center_csv):
    """
    Loads lung center coordinates from a CSV file and creates a mapping from image names to 3D coordinates.

    Args:
        center_csv (str): Path to the CSV file containing lung center coordinates.

    Returns:
        dict: A dictionary mapping image names (str) to their lung center coordinates as numpy arrays ([x, y, z]).
    """
    df = pd.read_csv(center_csv, dtype=str)
    image_to_center = {}
    for i in range(df.shape[0]):
        image_to_center[df.loc[i, 'image_name']] = np.array([float(df.loc[i, 'lung_center_world_x']),
                                                             float(df.loc[i, 'lung_center_world_y']),
                                                             float(df.loc[i, 'lung_center_world_z'])])
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
    circle_model,
    tokenizer,
    image_path,
    crop_center,
    text_list,
    device
):
    """
    Performs inference using a trained model on a given image and text list.

    Args:
        circle_model: The trained model to use for inference.
        tokenizer: The tokenizer for processing text inputs.
        image_path (str): Path to the input image file.
        crop_center: Center coordinates for cropping the image.
        text_list (list): List of texts to compare with the image.
        device: The device (CPU/GPU) to run the model on.

    Returns:
        tuple: A tuple containing:
            - clip_prob_list (list): List of probabilities from CLIP similarity.
            - cls_prob (numpy.ndarray): Classification probabilities.
            - visual_feature (numpy.ndarray): Extracted visual features.
    """
    # Move the model to the GPU and set evalation mode
    circle_model = circle_model.to(device)
    circle_model.eval()
    # Prepare cropped image for inference
    cropped_image = prepare_image(image_path, crop_center)

    with torch.no_grad():
        # Process image to get visual features and classification probabilities
        image_tensor_gpu = torch.unsqueeze(cropped_image, dim=0).to(device)
        cls_logits, visual_feature = circle_model.encode_image(image_tensor_gpu)
        cls_prob = nn.Sigmoid()(cls_logits)
        cls_prob = cls_prob.cpu().numpy()

        # Encode text inputs into features
        text_features = []
        for text in text_list:
            text_tokens = tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            text_feature = circle_model.encode_text(text_tokens)
            text_features.append(text_feature)
        if text_features:
            text_features = torch.stack(text_features, dim=0)

        # Calculate similarity between image and each text
        clip_prob_list = []
        for i in range(len(text_list)):
            similarity = circle_model.clip_forward(visual_feature, text_features[i])
            clip_prob = torch.sigmoid(similarity[0][0])
            clip_prob_list.append(clip_prob.detach().cpu().numpy())
    
    return clip_prob_list, cls_prob, visual_feature.detach().cpu()[0].numpy()


def encode_text(
    vision_encoder_dir,
    text_encoder_dir,
    text_list,
    device,
    num_classes=37
):
    """
    Loads a model and encodes text inputs into feature vectors.

    Args:
        vision_encoder_dir (str): Path to the vision encoder model directory.
        text_encoder_dir (str): Path to the text encoder model directory.
        text_list (dict): Dictionary mapping image names to text descriptions.
        device: The device (CPU/GPU) to run the model on.
        num_classes (int, optional): Number of output classes for the model. Defaults to 37.

    Returns:
        dict: A dictionary mapping image names to their encoded text features as numpy arrays.
    """
    # Load model and tokenizer
    circle_model, tokenizer = load_model(vision_encoder_dir, text_encoder_dir, num_classes)
    circle_model = circle_model.to(device)
    circle_model.eval()

    text_features = {}
    with torch.no_grad():
        # Process each text input and encode to feature vectors
        for image_name, text in text_list.items():
            text_tokens = tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            text_feature = circle_model.encode_text(text_tokens)
            text_features[image_name] = text_feature.detach().cpu()[0].numpy()

    return text_features

