import math
import numpy as np
import random

from math import cos
from math import sin
from torch.optim import AdamW, Adam


def separate_weight_decayable_params(params):
    """
    Split a list/iterable of parameters into two groups:
      - wd_params: parameters that should get weight decay (typically weight matrices)
      - no_wd_params: parameters that should NOT get weight decay (typically biases, LayerNorm/BatchNorm params, or 1-D vectors)
    Heuristic used here: parameters with ndim < 2 (i.e., scalars or vectors) are considered no-weight-decay.
    Args:
        params: iterable of torch.Tensor parameters
    Returns:
        (wd_params, no_wd_params): two lists of parameters
    Notes:
      - This is a simple heuristic. In many projects more explicit grouping is used (by name / module).
      - The function preserves the same parameter objects (does not clone).
    """
    wd_params, no_wd_params = [], []
    for param in params:
        # treat 1-D parameters (bias / LayerNorm scale) as no-weight-decay
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
    """
    Build and return an optimizer (AdamW by default; falls back to Adam if weight decay==0).
    Args:
      - params: iterable of parameters (or param groups) to optimize
      - lr: learning rate
      - wd: weight decay (L2 regularization) coefficient
      - betas, eps: Adam/AdamW optimizer hyperparameters
      - filter_by_requires_grad: if True, filter out parameters with requires_grad == False
      - group_wd_params: if True, group parameters into wd and no-wd groups using separate_weight_decayable_params
      - **kwargs: captured for forward compatibility (not used here)
    Returns:
      - torch.optim.Optimizer instance
    Behavior:
      - If wd == 0, returns torch.optim.Adam with no weight decay handling.
      - If group_wd_params == True and wd != 0, parameters are grouped, and AdamsW is created with a separate group having weight_decay=0.
    """
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        # If weight decay disabled, use Adam (no weight decay parameter)
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        # split params into decayable and non-decayable groups
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    # Use AdamW when weight decay requested (decoupled weight decay)
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def uniform_sample_point_from_unit_sphere(num_samples=1):
    """
    Uniformly sample points on the unit sphere surface.
    Sampling method:
      - Theta is sampled uniformly from [0, 2*pi)
      - Phi is sampled using inverse transform for cos(phi) to achieve uniform distribution on the sphere
    Args:
      - num_samples: number of 3D points to generate
    Returns:
      - numpy array shape (num_samples, 3) with dtype np.double
    Notes:
      - Each returned point has unit length (within numerical precision).
      - The function asserts num_samples >= 1.
    """
    assert num_samples >= 1, 'number of samples must be >= 1'

    sample_points = np.empty((num_samples, 3), dtype=np.double)
    for i in range(num_samples):
        # sample uniformly
        theta = 2 * math.pi * random.random()  # angle in XY plane
        # phi sampling via inverse CDF: cos(phi) ~ Uniform(-1, 1)
        phi = math.acos(2 * random.random() - 1.0)
        sample_points[i, 0] = math.cos(theta) * math.sin(phi)
        sample_points[i, 1] = math.sin(theta) * math.sin(phi)
        sample_points[i, 2] = math.cos(phi)

    return sample_points


def axis_angle_to_quaternion(axis, theta):
    """
    Convert axis-angle representation to quaternion (x, y, z, w).
    Args:
      - axis: 1-D numpy array of length 3 representing rotation axis (need not be normalized)
      - theta: rotation angle in radians
    Returns:
      - quaternion: numpy array shape (4,) in order [x, y, z, w], normalized
    Behavior & edge cases:
      - If axis has zero norm, the function returns identity quaternion [0,0,0,1] and prints a warning.
      - The returned quaternion is normalized.
    Notes:
      - Quaternion convention used here is (x, y, z, w) where w is the scalar part.
      - This quaternion represents a rotation of angle theta around the given axis.
    """
    assert axis.ndim == 1 and axis.size == 3

    # quaternion is in form of [x, y, z, w]
    quaternion = np.zeros(4, dtype=np.float32)
    quaternion[3] = 1  # default identity quaternion

    # normalize rotation axis to unit length
    norm = np.linalg.norm(axis)
    if norm == 0:
        print("Warning: axis is zero so return identity quaternion")
        return quaternion
    axis = axis / norm

    # compute quaternion components
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
    Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix.
    Args:
      - quaternion: 1-D numpy array of length 4 (x, y, z, w). Not required to be normalized.
    Returns:
      - rotation_matrix: 3x3 numpy array of dtype float32
    Behavior & edge cases:
      - If quaternion norm is zero, return identity matrix and print a warning.
      - The output rotation matrix satisfies R @ R.T = I (within numerical precision) after quaternion normalization.
    Formula:
      Uses the standard quaternion-to-matrix conversion with x,y,z as vector part and w as scalar part.
    """
    assert quaternion.ndim == 1 and quaternion.size == 4

    rotation_matrix = np.identity(3, dtype=np.float32)

    # normalize quaternion to unit length
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        print("Warning: quaternion is zero so return identity rotation matrix")
        return rotation_matrix
    quaternion = quaternion / norm

    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    # Fill rotation matrix entries using the standard conversion
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
    Convert axis-angle representation directly to 3x3 rotation matrix.
    Internally converts axis-angle -> quaternion -> rotation matrix.
    Args:
      - axis: 3-element numpy array axis
      - theta: rotation angle in radians
    Returns:
      - 3x3 numpy rotation matrix
    Notes:
      - This is a convenience wrapper leveraging the quaternion conversion.
      - Edge cases (zero axis) are handled by axis_angle_to_quaternion / quaternion_to_rotation_matrix.
    """
    quaternion = axis_angle_to_quaternion(axis, theta)

    return quaternion_to_rotation_matrix(quaternion)
