import argparse
import os
from importlib import import_module

import numpy as np
import torch
import torch.nn.functional as F

from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import ValPre


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TensorRT INT8 calibration data (npz) from val dataset")
    parser.add_argument(
        "--config",
        default="local_configs.NYUDepthv2.DFormerv2_S_Lite_mlp",
        help="Config module path for the deployed student model",
    )
    parser.add_argument("--out_dir", default="deploy/calib/nyu_384x512", help="Output directory for npz files")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=200)
    return parser.parse_args()


def build_val_dataset(cfg):
    data_setting = {
        "rgb_root": cfg.rgb_root_folder,
        "rgb_format": cfg.rgb_format,
        "gt_root": cfg.gt_root_folder,
        "gt_format": cfg.gt_format,
        "transform_gt": cfg.gt_transform,
        "x_root": cfg.x_root_folder,
        "x_format": cfg.x_format,
        "x_single_channel": cfg.x_is_single_channel,
        "class_names": cfg.class_names,
        "train_source": cfg.train_source,
        "eval_source": cfg.eval_source,
        "dataset_name": cfg.dataset_name,
        "backbone": cfg.backbone,
    }
    preprocess = ValPre(cfg.norm_mean, cfg.norm_std, cfg.x_is_single_channel, cfg)
    return RGBXDataset(data_setting, "val", preprocess)


def resize_inputs(rgb, depth, height, width):
    rgb = F.interpolate(rgb.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    depth = F.interpolate(depth.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    return rgb, depth


def main():
    args = parse_args()
    cfg = getattr(import_module(args.config), "C")
    cfg.pad = False
    dataset = build_val_dataset(cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    total = min(args.num_samples, len(dataset))

    index_lines = []
    for i in range(total):
        sample = dataset[i]
        rgb = sample["data"]
        depth = sample["modal_x"]
        rgb, depth = resize_inputs(rgb, depth, args.height, args.width)

        rgb_np = rgb.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        depth_np = depth.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        fn = sample.get("fn", f"{i:06d}")
        safe = os.path.basename(str(fn)).replace("/", "_")
        out_path = os.path.join(args.out_dir, f"{i:06d}_{safe}.npz")
        np.savez_compressed(out_path, rgb=rgb_np, depth=depth_np)
        index_lines.append(out_path)

    index_path = os.path.join(args.out_dir, "index.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        for p in index_lines:
            f.write(p + "\n")

    print("out_dir:", args.out_dir)
    print("num_samples:", total)
    print("shape:", f"rgb/depth = 3x{args.height}x{args.width}")
    print("index:", index_path)


if __name__ == "__main__":
    main()

