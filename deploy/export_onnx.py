import argparse
import os
from importlib import import_module

import torch
import torch.nn as nn

from models.builder import EncoderDecoder as SegModel
from utils.engine.logger import get_logger
from utils.pyt_utils import load_model


class DeployWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, depth):
        return self.model(rgb, depth)


def parse_args():
    parser = argparse.ArgumentParser(description="Export DFormer/DFormerv2 model to ONNX")
    parser.add_argument(
        "--config",
        default="local_configs.NYUDepthv2.DFormerv2_S_Lite_mlp",
        help="Config module path, e.g. local_configs.NYUDepthv2.DFormerv2_S_Lite_mlp",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Supports normal training checkpoint and distill checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="deploy/onnx/dformerv2_s_lite_mlp.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--height", type=int, default=384, help="Fixed input height")
    parser.add_argument("--width", type=int, default=512, help="Fixed input width")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for export. CUDA is recommended if available.",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a dry forward pass before export",
    )
    return parser.parse_args()


def build_model(config_module, device):
    cfg = getattr(import_module(config_module), "C")
    model = SegModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    model.to(device)
    model.eval()
    return cfg, model


def maybe_check_onnx(output_path, logger):
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX checker passed")
    except ImportError:
        logger.warning("onnx is not installed, skip ONNX checker")


def main():
    args = parse_args()
    logger = get_logger()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, fallback to CPU export")
        args.device = "cpu"

    device = torch.device(args.device)
    cfg, model = build_model(args.config, device)

    logger.info(f"loading checkpoint from {args.checkpoint}")
    load_model(model, args.checkpoint, is_restore=False)
    model.eval()

    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    depth = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    if args.verify:
        with torch.no_grad():
            logits = model(rgb, depth)
        logger.info(f"dry forward output shape: {tuple(logits.shape)}")

    wrapper = DeployWrapper(model).to(device).eval()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "export ONNX with config=%s, backbone=%s, input=%dx3x%dx%d",
        args.config,
        cfg.backbone,
        args.batch_size,
        args.height,
        args.width,
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (rgb, depth),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["rgb", "depth"],
            output_names=["logits"],
            dynamic_axes=None,
        )

    logger.info(f"exported ONNX to {args.output}")
    maybe_check_onnx(args.output, logger)


if __name__ == "__main__":
    main()
