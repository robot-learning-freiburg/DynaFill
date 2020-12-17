
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import flow
from models.gan.ours.coarse_resnet import Inpainting
from models.gan.ours.refinement_resnet import TranslationSimple2
from models.depth.self_supervised_sparse_to_dense import DepthCompletionNet
from utils import dotdict
from datasets import DynaFillTrajectory

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_split_dir",
        type=str,
        help="Path to train/ or validation/ directory of DynaFill dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to run inference"
    )
    return parser.parse_args()

def warp(rgb_previous, depth_previous, pose_previous, pose_current, K, K_inv, camera_local):
    R, t = flow.camera_transform(pose_previous, pose_current, camera_local, forward=True)

    R = R.to(depth_previous.device)
    t = t.to(depth_previous.device)
    K = K.to(depth_previous.device)
    K_inv = K_inv.to(depth_previous.device)

    # Warping: (t - 1) -> t
    egoflow, pixel_coords = flow.calculate_egomotion_flow(R, t, depth_previous, K, K_inv) # [B, 2, W, H]
    rgb_previous_warped, _ = flow.warp_image(rgb_previous.to(pixel_coords.device), pixel_coords)

    return rgb_previous_warped

def to_jet_colormap(d): # [B, 1, H, W] in [0, MAX_DEPTH]
    if d.dim() == 4:
        d = d[0]

    d_jet = d.detach()
    d_jet = torch.ones_like(d) + (torch.log(d_jet / 1000) / 7)
    d_jet = torch.clamp(d_jet, 0.0, 1.0)
    d_jet = cv2.applyColorMap((d_jet.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)

    return d_jet

def forward(M, rgb_dynamic, depth_dynamic, mask, pose, K, K_inv, camera_local):
    rgb_inpainted = dict()
    rgb_inpainted_coarse = dict()
    depth_inpainted = dict()

    T = rgb_dynamic.size(1)
    for t in range(T):

        # Housekeeping
        if t - 2 in rgb_inpainted and t - 2 != 0:
            del rgb_inpainted[t - 2]
        if t - 2 in depth_inpainted and t - 2 != 0:
            del depth_inpainted[t - 2]

        # t
        rgb_dynamic_t = rgb_dynamic[:, t]
        depth_dynamic_t = depth_dynamic[:, t] # * MAX_DEPTH
        mask_t = mask[:, t]
        pose_t = pose[:, t]

        # t - 1
        mask_t_1 = mask[:, max(t - 1, 0)]
        pose_t_1 = pose[:, max(t - 1, 0)]

        # -------------------------------------- Forward pass --------------------------------------

        if mask_t.max().item() == 0:
            rgb_inpainted_t_ = rgb_dynamic_t
            rgb_inpainted_t = rgb_dynamic_t
            print("Forward pass not needed.")
        else:
            # RGB Inpainting
            # - Coarse
            rgb_inpainted_t_ = M.coarse_inpainting(rgb_dynamic_t, mask_t)

            # t - 1
            if t >= 1:
                rgb_inpainted_t_1 = rgb_inpainted[t - 1].detach()
                depth_inpainted_t_1 = depth_inpainted[t - 1].detach()
                rgb_inpainted_t_1_warped_to_t = warp(
                    rgb_inpainted_t_1,
                    depth_inpainted_t_1,
                    pose_t_1, pose_t,
                    K, K_inv, camera_local
                )
            else:
                rgb_inpainted_t_1_warped_to_t = rgb_inpainted_t_.detach()

            # - Refinement
            refinement_input = rgb_inpainted_t_
            rgb_inpainted_t = M.refined_translation(refinement_input, mask_t, rgb_inpainted_t_1_warped_to_t)

        rgb_inpainted_coarse[t] = rgb_inpainted_t_.detach()
        rgb_inpainted[t] = rgb_inpainted_t.detach()

        # Depth Completion (t)
        depth_completion_input_t = rgb_inpainted_t.detach()
        depth_inpainted_t = M.depth_completion(dict(
            rgb=(depth_completion_input_t + 1) * 127.5, # [0, 255]
            d=depth_dynamic_t * (1 - mask_t) # [0, MAX_DEPTH]
        ))

        # - Inpaint (copy the existing pixels)
        depth_dynamic_t = depth_dynamic_t.to(depth_inpainted_t.device)
        depth_inpainted_t = depth_inpainted_t * mask_t + depth_dynamic_t * (1 - mask_t)

        depth_inpainted[t] = depth_inpainted_t.detach()

        yield dict(
            rgb_dynamic_t=rgb_dynamic_t,
            rgb_inpainted_t_=(rgb_inpainted_t * mask_t + rgb_dynamic_t * (1 - mask_t)),
            rgb_inpainted_t=rgb_inpainted_t,
            pose_t=pose_t,
            depth_dynamic_t=depth_dynamic_t,
            depth_inpainted_t=depth_inpainted_t,
            mask_t=mask_t
        )

def load_models():
    print("Instantiating models...")
    models = dict(
        coarse_inpainting=Inpainting(
            input_channels=4,
            base_channels=32
        ),
        refined_translation=TranslationSimple2(
            base_channels=32,
            n_warped_imgs=1,
            range_rgb=[-1, 1]
        ),
        depth_completion=DepthCompletionNet(
            args=dotdict(dict(
                input="rgbd",
                layers=18
            ))
        )
    )

    for model_name, model in models.items():
        state_dict = torch.load(CHECKPOINTS_DIR / f"{model_name}.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded state dict for {model_name}.")

    return models

def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    models = dotdict(load_models())

    data = DynaFillTrajectory(args.dataset_split_dir)
    K = data.K
    K_inv = K.inverse()
    camera_local = data.camera_local

    loader = DataLoader(data, batch_size=1, num_workers=1, shuffle=False)
    print("Loaded dataset. Size:", len(data))

    for trajectory in tqdm(loader):

        rgb_dynamic = trajectory["rgb"].to(args.device)
        depth_dynamic = trajectory["depth"].to(args.device)
        mask = trajectory["mask"].to(args.device)
        pose = trajectory["pose"].to(args.device)

        T = rgb_dynamic.size(1)

        for output in tqdm(forward(models, rgb_dynamic, depth_dynamic, mask, pose, K, K_inv, camera_local), total=T):
            rgb_inpainted_t_ = (output["rgb_inpainted_t_"] + 1) / 2
            rgb_inpainted_t = (output["rgb_inpainted_t"] + 1) / 2
            rgb_dynamic_t = (output["rgb_dynamic_t"] + 1) / 2
            depth_dynamic_t = output["depth_dynamic_t"]
            depth_inpainted_t = output["depth_inpainted_t"]
            mask_t = output["mask_t"]

            # RGB
            rgb_dynamic_t = (rgb_dynamic_t[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()[:, :, ::-1]
            # rgb_inpainted_t_ = (rgb_inpainted_t_[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()[:, :, ::-1]
            rgb_inpainted_t = (rgb_inpainted_t[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()[:, :, ::-1]

            # Depth
            depth_dynamic_t = to_jet_colormap(depth_dynamic_t)[:, :, ::-1]
            depth_inpainted_t = to_jet_colormap(depth_inpainted_t)[:, :, ::-1]

            # Mask
            mask_t = (mask_t[0][0] * 255).round().byte().cpu().numpy()

            cv2.imshow("RGB Input", rgb_dynamic_t)
            cv2.imshow("Mask", mask_t)
            cv2.imshow("RGB Output", rgb_inpainted_t)
            cv2.imshow("Depth Input", depth_dynamic_t)
            cv2.imshow("Depth Output", depth_inpainted_t)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()

