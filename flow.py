

import numpy as np
import torch
import torch.nn.functional as F
from transformations import euler_matrix


def deg2rad(deg):
    return deg / 180 * np.pi

def generate_pixel_grid(h=512, w=512):
    """
    Generates a grid of pixel coordinates.
    (W, H)
    """

    xx, yy = torch.meshgrid(
        torch.arange(w, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32)
    )
    zz = torch.ones(w, h)
    return torch.stack([xx, yy, zz])

def to_camera_frame(pose_matrix):
    """
    Transforms the pose from the standard right-handed
    coordinate system (x forward )into the camera
    coordinate system (z forward).
    """
    batch = (pose_matrix.dim() == 3)

    # Remember the old coordinate axes
    if not batch:
        i, j, k = pose_matrix[:3, :3].clone()
        x, y, z = pose_matrix[:3, 3].clone()

    # 0. Translate to origin
    pose_matrix = pose_matrix.clone()
    if batch:
        t = pose_matrix[:, :4, 3].clone().unsqueeze(-1)
        pose_matrix[:, :3, 3] = 0
    else:
        t = pose_matrix[:4, 3].clone()
        pose_matrix[:3, 3] = 0

    # 1. Yaw: apply RH -90 degrees
    R_yaw = torch.Tensor(euler_matrix(0, 0, deg2rad(90)))
    pose_matrix = R_yaw @ pose_matrix
    t = R_yaw @ t

    # 2. Roll: apply RH -90 degrees
    R_roll = torch.Tensor(euler_matrix(deg2rad(90), 0, 0))
    pose_matrix = R_roll @ pose_matrix
    t = R_roll @ t

    # 4. Translate back
    if batch:
        pose_matrix[:, :3, 3] = t[:, :3].squeeze(-1)
    else:
        pose_matrix[:3, 3] = t[:3]

    # Check if everything is fine
    if not batch:
        i_, j_, k_ = pose_matrix[:3, :3]
        x_, y_, z_ = pose_matrix[:3, 3]

        assert torch.allclose(i, k_), (i, k_)
        assert torch.allclose(j, -i_), (j, -i_)
        assert torch.allclose(k, -j_), (k, -j_)

    return pose_matrix

def unreal_to_right_pose(pose):
    """
    Transform the pose from Unreal coordinate
    system to the right-hand coordinate system.

    pose: x, y, z, roll, pitch, yaw (6,) or (batch, 6)
    """
    # batch = (pose.dim() == 2)

    transform = torch.Tensor([
         1, # x
        -1, # y
         1, # z
         1, # roll
        -1, # pitch
        -1  # yaw -> Unreal uses left rotation for yaw
    ])

    # if batch:
    #     transform = transform.unsqueeze(0)

    return pose * transform

def pose_to_matrix(pose):
    """
    Builds a homogeneous pose matrix out of
    the pose vector.
    (4, 4)

    pose: x, y, z, roll, pitch, yaw (6,) or (B, 6)
    """
    # Asserts
    assert pose.dim() in {1, 2}
    if pose.dim() == 1:
        assert pose.size(0) == 6
    elif pose.dim() == 2:
        assert pose.size(1) == 6

    # If it is a batch, do it recursively
    if pose.dim() == 2:
        poses = pose
        return torch.stack(tuple(map(pose_to_matrix, poses)), dim=0) # (B, 4, 4)

    pose = pose.clone()
    roll, pitch, yaw = deg2rad(pose[3:])
    R = euler_matrix(roll, pitch, yaw)
    P = R.copy()
    P[:3, 3] = pose[:3]
    return torch.tensor(P, dtype=torch.float32) # (4, 4)

def camera_transform(pose0, pose1, camera_local, forward=True):
    """
    Calculates the transform between two poses in the
    camera frame: R and t.

    forward: pose0 -> pose1

    pose0: x, y, z, roll, pitch, yaw (6,) or (B, 6) in meters and degrees in Unreal frame
    pose1: x, y, z, roll, pitch, yaw (6,) or (B, 6) in meters and degrees in Unreal frame
    camera_local: x, y, z, roll, pitch, yaw (6,)
    """
    batch = (pose0.dim() == 2)

    # Forward or backward transform?
    if not forward:
        return camera_transform(pose1, pose0, camera_local)

    # Convert weird Unreal coordinate system to the right coordinate system
    pose0 = unreal_to_right_pose(pose0)
    pose1 = unreal_to_right_pose(pose1)
    camera_local = unreal_to_right_pose(camera_local)

    # Build pose matrices
    P0 = pose_to_matrix(pose0)
    P1 = pose_to_matrix(pose1)
    C = pose_to_matrix(camera_local)

    C0 = P0 @ C # Global pose of the previous camera # (4 x 4) * (4 x 4)
    C1 = P1 @ C # Global pose of the current camera

    C0_local = torch.eye(4) # Previous camera pose in the previous camera's frame (identity)
    C1_local = C0.inverse() @ C1 # Current camera pose in the previous camera's frame

    # Move cameras into camera coordinate system
    C0_local = to_camera_frame(C0_local)
    C1_local = to_camera_frame(C1_local)

    # Get the transform from C0 to C1
    M = C1_local @ C0_local.inverse()

    if batch:
        t = M[:, :4, 3].clone()
        R = M.clone()
        R[:, :3, 3] = 0
        return R[:, :-1, :-1], t[:, :-1]
    else:
        t = M[:4, 3].clone()
        R = M.clone()
        R[:3, 3] = 0
        return R[:-1, :-1], t[:-1]


def calculate_egomotion_flow(R, t, z, K, K_inv=None):
    """
    Calculates the optical flow induced by ego-motion.

    Arguments (all torch.Tensors):
    R: rotation matrix (3, 3) or (B, 3, 3)
    t: translation (3,) or (B, 3)
    z: current depth (1, H, W) in meters or (B, 1, H, W)
    K: camera intrinsic matrix (3, 3)
    K_inv (optional): inverse of camera instrinsic matrix (3, 3) or (B, 3, 3)
    """
    batch = (R.dim() == 3)

    R = R.to(z.device)
    t = t.to(z.device)

    # Get height and width
    if z.dim() == 4:
        _, _, H, W = z.size()
    elif z.dim() == 3:
        _, H, W = z.size()

    # Calculate the inverse if not provided
    if K_inv is None:
        K_inv = K.inverse()

    # Generate a grid of pixels coordinates
    X = generate_pixel_grid(H, W).to(z.device) # [3, W, H]

    # Pixel rotation
    pre_C = K @ R @ K_inv

    # Pixel translation
    if batch:
        B = R.size(0)
        T = torch.ones(B, 3, W, H).to(z.device) * t.view(-1, 3, 1, 1)
    else:
        T = torch.ones(3, W, H).to(z.device) * t.view(3, 1, 1)

    z = torch.clamp(z, min=1e-12) # Avoid division by 0
    l = T / (z.transpose(1, 2) if not batch else z.transpose(2, 3)) # [1, W, H]

    # Transform the pixels
    X_ = torch.tensordot(pre_C, X, dims=1)
    if batch:
        if K.dim() == 3: # Intrinsics also provided in a batch [B, 3, 3]
            B, _, W, H  = l.size()
            l = l.view(B, 3, -1) # [B, 3, W * H]
            Kl = K @ l # [B, 3, W * H]
            X_ = X_ + Kl.view(B, 3, W, H)
        else:
            X_ = X_ + torch.tensordot(K, l, dims=([1], [1])).transpose(0, 1)
    else:
        X_ = X_ + torch.tensordot(K, l, dims=1)

    # Pixels: homogenous -> Cartesian
    if batch:
        denominator = X_[:, 2, None, :, :]
    else:
        denominator = X_[2, :, :]

    denominator[denominator == 0.] = denominator[denominator == 0.] + 1e-8
    X_ = X_ / denominator

    # Calculate the optical flow
    egoflow = X_ - X

    # Optical flow: remove the homogenous channel
    if batch:
        egoflow = egoflow[:, :2, :, :]
        return egoflow, X_[:, :2, :, :] # [2, W, H]
    else:
        egoflow = egoflow[:2, :, :]
        return egoflow, X_[:2, :, :] # [2, W, H]


def warp_image(image, pixels, mode="bilinear", padding="zeros", eps=0.2):
    """
    Warps the given image using the specified pixel coordinates.

    image: (3, H, W) or (?, 3, H, W)
    pixels: (2, W, H) or (?, 2, W, H)
    """

    squeeze_later = False
    if image.dim() == 3:
        squeeze_later = True
        image = image.unsqueeze(0)
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)

    # [0, SIZE] -> [-1, 1]
    _, _, H, W = image.size()
    normalization = torch.FloatTensor([2 / (W - 1), 2 / (H - 1)]).unsqueeze(1).unsqueeze(2).to(pixels.device)
    pixels = pixels * normalization - 1
    pixels = torch.clamp(pixels, min=-1. - eps, max=1 + eps)

    # [1, 2, W, H] -> [1, H, W, 2]
    pixels = pixels.permute(0, 3, 2, 1)

    warped = F.grid_sample(image, pixels.to(image.device), mode=mode, padding_mode=padding)
    mask = pixels.ge(-1) & pixels.le(1)
    mask = (mask[:, :, :, 0] & mask[:, :, :, 1]).float().unsqueeze(1)

    if squeeze_later:
        warped = warped.squeeze(0)
        mask = mask.squeeze(0) if mask is not None else None

    return warped, mask
