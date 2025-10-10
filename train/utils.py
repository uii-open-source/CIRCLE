import math
import numpy as np
import random

from math import cos
from math import sin
from torch.optim import AdamW, Adam


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-4,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=False,
    group_wd_params=True,
    **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def uniform_sample_point_from_unit_sphere(num_samples=1):
    """
    uniformly sample a point from a unit sphere
    If the number of samples is large, this sampling strategy ensures the uniformity of point distribution on sphere
    :return: a list of 3D points (np.array)
    """
    assert num_samples >= 1, 'number of samples must be >= 1'

    sample_points = np.empty((num_samples, 3), dtype=np.double)
    for i in range(num_samples):
        theta = 2 * math.pi * random.random()
        phi = math.acos(2 * random.random() - 1.0)
        sample_points[i, 0] = math.cos(theta) * math.sin(phi)
        sample_points[i, 1] = math.sin(theta) * math.sin(phi)
        sample_points[i, 2] = math.cos(phi)

    return sample_points


def axis_angle_to_quaternion(axis, theta):
    """
    Compute 4-d quaternion (x, y, z, w) from angle axis form.
    :param          axis: 3-D numpy array of the rotation axis which does not
                    have to be normalized.
    :param          theta: rotation angle in radian
    :return:        quaternion: 4-D numpy array in terms of (x,y,z,w) which has
                    been normalized.
    """
    assert axis.ndim == 1 and axis.size == 3

    # quaternion is in form of [x, y, z, w]
    quaternion = np.zeros(4, dtype=np.float32)
    quaternion[3] = 1

    # normalize rotation axis
    norm = np.linalg.norm(axis)
    if norm == 0:
        print("Warning: axis is zero so return identity quaternion")
        return quaternion
    axis = axis / norm

    cos_theta_2 = cos(theta/2)
    sin_theta_2 = sin(theta/2)
    w = cos_theta_2
    x = axis[0] * sin_theta_2
    y = axis[1] * sin_theta_2
    z = axis[2] * sin_theta_2

    quaternion[0] = x
    quaternion[1] = y
    quaternion[2] = z
    quaternion[3] = w

    return quaternion


def quaternion_to_rotation_matrix(quaternion):
    """
    Compute 3x3 rotation matrix R from 4-D quaternion (x,y,z,w). R * v = v'
    :param          quaternion: 4-D numpy array in terms of (x,y,z,w) which does
                    not have to be normalized.
    :return:        matrix: 3x3 rotation matrix satisfying R * R' = I.
    """
    assert quaternion.ndim == 1 and quaternion.size == 4

    rotation_matrix = np.identity(3, dtype=np.float32)

    # normalize quaternion
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        print("Warning: quaternion is zero so return identity rotation matrix")
        return rotation_matrix
    quaternion = quaternion / norm

    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    rotation_matrix[0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[0, 1] = 2 * (x * y - z * w)
    rotation_matrix[0, 2] = 2 * (x * z + y * w)

    rotation_matrix[1, 0] = 2 * (x * y + z * w)
    rotation_matrix[1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[1, 2] = 2 * (y * z - x * w)

    rotation_matrix[2, 0] = 2 * (x * z - y * w)
    rotation_matrix[2, 1] = 2 * (y * z + x * w)
    rotation_matrix[2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrix


def axis_angle_to_rotation_matrix(axis, theta):
    """
    Compute 3x3 rotation matrix R from angle axis form.
    :param          axis: 3-D numpy array of the rotation axis which does not
                    have to  have to be normalized.
    :param          theta: rotation angle in radian
    :return:        matrix: 3x3 rotation matrix satisfying R * R' = I.
    """
    quaternion = axis_angle_to_quaternion(axis, theta)

    return quaternion_to_rotation_matrix(quaternion)
