import math
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk  # SimpleITK for medical image I/O and processing
import torch
from torch.utils.data import Dataset

from train.utils import axis_angle_to_rotation_matrix, uniform_sample_point_from_unit_sphere
# utility functions:
# axis_angle_to_rotation_matrix: convert axis-angle representation to rotation matrix
# uniform_sample_point_from_unit_sphere: sample random point on a unit sphere for rotation axis


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
        half_len = (crop_size[dim]) * spacing[dim] / 2.0
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


class CIRCLEDataset(Dataset):
    """
    PyTorch Dataset for CIRCLE project.
    Loads CT images, labels, lung center coordinates, and report text.
    Applies cropping, normalization, and augmentation.
    """
    def __init__(self, data_folder, label_csv, lung_center_csv, report_csv):
        self.data_folder = data_folder
        # Load mapping from image name to label vector
        self.image_to_label = self.load_image_to_label(label_csv)
        # Load lung center coordinates for each image
        self.image_to_center = self.get_lung_center(lung_center_csv)
        # Load report text (description + conclusion) for each image
        self.image_to_text = self.load_report_text(report_csv)
        # Prepare list of samples for __getitem__
        self.samples = self.prepare_samples()
        print('--------------- num of sample: {} ----------------'.format(len(self.samples)))

    def get_lung_center(self, center_csv):
        """
        Read lung center coordinates from CSV
        Args:
            center_csv: path to CSV containing 'image_name' and lung_center_world_x/y/z
        Returns:
            dict: image_name -> np.array([x, y, z])
        """
        df = pd.read_csv(center_csv, dtype={'image_name': str})
        image_to_center = {}
        for i in range(df.shape[0]):
            image_to_center[df.loc[i, 'image_name']] = np.array([df.loc[i, 'lung_center_world_x'],
                                                                 df.loc[i, 'lung_center_world_y'],
                                                                 df.loc[i, 'lung_center_world_z']])
        return image_to_center

    def load_image_to_label(self, label_csv):
        """
        Read image labels from CSV
        Args:
            label_csv: path to CSV containing image_name and 37 label columns
        Returns:
            dict: image_name -> list of 37 labels
        """
        image_to_label = {}
        df = pd.read_csv(label_csv, dtype={'image_name': str})
        # skip first columns, next 37 columns are labels
        sorted_cols = df.columns.tolist()[1:]
        assert len(sorted_cols) == 37
        for i in range(df.shape[0]):
            image_to_label[df.loc[i, 'image_name']] = [df.loc[i, col] for col in sorted_cols]
        return image_to_label

    def load_report_text(self, csv_file):
        """
        Load report text (description + conclusion) from CSV
        Returns:
            dict: image_name -> (description, conclusion)
        """
        df = pd.read_csv(csv_file, dtype={'image_name': str})
        image_to_text = {}
        for i in range(df.shape[0]):
            image_name = df.loc[i, 'image_name']
            image_to_text[image_name] = df.loc[i, 'finding'], df.loc[i, 'impression']
        return image_to_text

    def prepare_samples(self):
        """
        Prepare list of samples containing:
        (image_path, report_text, lung_center, labels)
        """
        samples = []
        image_names = sorted(self.image_to_text.keys())
        img_dir = self.data_folder
        for name in image_names:
            if name not in self.image_to_label or name not in self.image_to_center:
                continue
            image_path = os.path.join(img_dir, name, 'CT.nii.gz')
            if not os.path.exists(image_path):
                print('{} not exist'.format(image_path))
                continue
            description, conclusion = self.image_to_text[name]
            report = description + conclusion
            lung_center = self.image_to_label[name]  # might be replaced with actual lung center if needed
            label = self.image_to_label[name]
            samples.append((image_path, report, lung_center, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def prepare_image(self, img_path, crop_center):
        """
        Load image, apply cropping, optional augmentation, and normalization
        Args:
            img_path: path to CT image (.nii.gz)
            crop_center: 3D coordinates of center for cropping
        Returns:
            torch.FloatTensor of shape (1, D, H, W)
        """
        # Augmentation probabilities and parameters
        rot_prob = 0
        rot_angle_degree = 10

        scale_prob = 0
        scale_isotropic = True
        scale_min_ratio = 0.95
        scale_max_ratio = 1.05

        shift_prob = 0
        shift_mm = 20.0

        # Normalization
        mean = -400
        stddev = 600
        clip = False

        # Crop parameters
        crop_size = np.array([300, 300, 160])  # voxels
        crop_spacing = np.array([1.0, 1.0, 2.0])  # mm
        crop_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)
        crop_scale_ratio = np.array([1.0, 1.0, 1.0])

        # Randomly decide whether to apply augmentations
        rotate_flag = np.random.choice([False, True], p=[1 - rot_prob, rot_prob]) if 0 <= rot_prob <= 1 else False
        scale_flag = np.random.choice([False, True], p=[1 - scale_prob, scale_prob]) if 0 <= scale_prob <= 1 else False
        shift_flag = np.random.choice([False, True], p=[1 - shift_prob, shift_prob]) if 0 <= shift_prob <= 1 else False

        if rotate_flag or scale_flag or shift_flag:
            if rotate_flag:
                # Random rotation around a unit sphere axis
                rot_axis = uniform_sample_point_from_unit_sphere()
                rot_axis = rot_axis[0]
                angle = np.random.random() * rot_angle_degree * math.pi / 180.0
                crop_axes = axis_angle_to_rotation_matrix(rot_axis, angle)
            if scale_flag:
                # Random isotropic or anisotropic scaling
                if scale_isotropic:
                    scale_ratio = np.random.uniform(scale_min_ratio, scale_max_ratio)
                    scale_ratio = np.array([scale_ratio] * 3)
                else:
                    scale_ratio = np.random.uniform(scale_min_ratio, scale_max_ratio, (3,))
                crop_scale_ratio *= scale_ratio
                crop_spacing = crop_spacing / crop_scale_ratio
            if shift_flag:
                # Random shift of crop center
                shift = np.random.uniform(-shift_mm, shift_mm, (3,))
                crop_center += shift

        # Load image using SimpleITK
        img = sitk.ReadImage(img_path)
        # Crop and resample image
        cropped_img = crop_image(img, crop_center, crop_spacing, crop_size, crop_axes)
        # Convert to numpy array
        cropped_img = sitk.GetArrayFromImage(cropped_img)
        # Intensity normalization
        cropped_img = intensity_normalize(cropped_img, mean, stddev, clip)

        # Convert to torch tensor and add channel dimension
        cropped_img = torch.from_numpy(cropped_img)
        cropped_img = torch.unsqueeze(cropped_img, 0)  # (1, D, H, W)
        cropped_img = cropped_img.float()
        return cropped_img

    def __getitem__(self, index):
        """
        Return one sample for PyTorch DataLoader
        Returns:
            image_tensor: torch.FloatTensor (1, D, H, W)
            input_text: report text (description + conclusion)
            labels: torch.Tensor of 37 labels
        """
        image_path, input_text, lung_center, labels = self.samples[index]
        labels = torch.Tensor(labels)
        image_tensor = self.prepare_image(image_path, lung_center)
        return image_tensor, input_text, labels
