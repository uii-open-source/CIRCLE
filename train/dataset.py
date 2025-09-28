import math
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from md import Frame3d
from md.mdmath.python.math_tools import uniform_sample_point_from_unit_sphere
from md.mdmath.python.rotation3d import axis_angle_to_rotation_matrix
import md.image3d.python.image3d_io as cio
import md.image3d.python.image3d_tools as ctools


def crop_image(im, crop_center, crop_spacing, crop_size, crop_axes=None):
    frame = Frame3d()
    frame.set_origin(crop_center)
    frame.set_spacing(crop_spacing)
    if crop_axes is None:
        frame.set_axes_to_rai()
    else:
        frame.set_axes(crop_axes)

    crop_size = np.array(crop_size)
    crop_origin = frame.voxel_to_world(-crop_size / 2.0)
    frame.set_origin(crop_origin)

    crop = ctools.resample_trilinear(im, frame, crop_size, default_value=-1024)
    return crop


class CIRCLEDataset(Dataset):
    def __init__(self, data_folder, label_csv, lung_center_csv, report_csv):
        self.data_folder = data_folder
        self.image_to_label = self.load_image_to_label(label_csv)
        self.image_to_center = self.get_lung_center(lung_center_csv)
        self.image_to_text = self.load_report_text(report_csv)
        self.samples = self.prepare_samples()
        print('--------------- {} ----------------'.format(len(self.samples)))

    def get_lung_center(self, center_csv):
        df = pd.read_csv(center_csv)
        image_to_center = {}
        for i in range(df.shape[0]):
            image_to_center[df.loc[i, 'image_name']] = np.array([df.loc[i, 'lung_center_world_x'],
                                                                 df.loc[i, 'lung_center_world_y'],
                                                                 df.loc[i, 'lung_center_world_z']])
        return image_to_center

    def load_image_to_label(self, label_csv):
        image_to_label = {}
        df = pd.read_csv(label_csv)
        # TODO: hard code
        sorted_cols = df.columns.tolist()[3:]
        assert len(sorted_cols) == 37
        for i in range(df.shape[0]):
            image_to_label[df.loc[i, 'image_name']] = [df.loc[i, col] for col in sorted_cols]
        return image_to_label

    def load_report_text(self, csv_file):
        df = pd.read_csv(csv_file)
        image_to_text = {}
        for i in range(df.shape[0]):
            image_name = df.loc[i, 'image_name']
            image_to_text[image_name] = df.loc[i, 'description'], df.loc[i, 'conclusion']
        return image_to_text

    def prepare_samples(self):
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
            lung_center = self.image_to_label[name]
            label = self.image_to_label[name]
            samples.append((image_path, report, lung_center, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def prepare_image(self, img_path, crop_center):
        rot_prob = 0.5
        rot_angle_degree = 10

        scale_prob = 0.5
        scale_isotropic = True
        scale_min_ratio = 0.95
        scale_max_ratio = 1.05

        shift_prob = 0.5
        shift_mm = 20.0

        mean = 0
        stddev = 1000
        clip = False

        crop_size = np.array([300, 300, 160])
        crop_spacing = np.array([1.0, 1.0, 2.0])
        img = cio.read_image(img_path, dtype=np.float32)

        crop_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)
        crop_scale_ratio = np.array([1.0, 1.0, 1.0])
        rotate_flag = np.random.choice([False, True], p=[1 - rot_prob, rot_prob]) if 0 <= rot_prob <= 1 else False
        scale_flag = np.random.choice([False, True], p=[1 - scale_prob, scale_prob]) if 0 <= scale_prob <= 1 else False
        shift_flag = np.random.choice([False, True], p=[1 - shift_prob, shift_prob]) if 0 <= shift_prob <= 1 else False
        if rotate_flag or scale_flag or shift_flag:
            if rotate_flag:
                # random rotate input image in degrees
                rot_axis = uniform_sample_point_from_unit_sphere()
                rot_axis = rot_axis[0]
                angle = np.random.random() * rot_angle_degree * math.pi / 180.0
                crop_axes = axis_angle_to_rotation_matrix(rot_axis, angle)
            if scale_flag:
                # random scale input image with scale ratio
                if scale_isotropic:
                    scale_ratio = np.random.uniform(scale_min_ratio, scale_max_ratio)
                    scale_ratio = np.array([scale_ratio] * 3)
                else:
                    scale_ratio = np.random.uniform(scale_min_ratio, scale_max_ratio, (3,))
                crop_scale_ratio *= scale_ratio
                crop_spacing = crop_spacing / crop_scale_ratio
            if shift_flag:
                shift = np.random.uniform(-shift_mm, shift_mm, (3,))
                crop_center += shift
        cropped_img = crop_image(img, crop_center, crop_spacing, crop_size, crop_axes)
        ctools.intensity_normalize(cropped_img, mean, stddev, clip)

        cropped_img = torch.from_numpy(cropped_img.to_numpy())
        cropped_img = torch.unsqueeze(cropped_img, 0)
        cropped_img = cropped_img.float()
        return cropped_img

    def __getitem__(self, index):
        image_path, input_text, lung_center, labels = self.samples[index]
        labels = torch.Tensor(labels)
        image_tensor = self.prepare_image(image_path, lung_center)
        return image_tensor, input_text, labels
