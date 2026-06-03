import argparse
import os
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import EncoderDecoder as SegModel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import ValPre
from utils.engine.logger import get_logger
from utils.pyt_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Validate output alignment across PyTorch / ONNX / TensorRT")
    parser.add_argument(
        "--config",
        default="local_configs.NYUDepthv2.DFormerv2_S_Lite_mlp",
        help="Config module path for the deployed student model",
    )
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path")
    parser.add_argument("--onnx", default=None, help="ONNX model path")
    parser.add_argument("--engine", default=None, help="TensorRT engine path")
    parser.add_argument("--height", type=int, default=384, help="Validation input height")
    parser.add_argument("--width", type=int, default=512, help="Validation input width")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of validation samples to compare")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="PyTorch reference device")
    parser.add_argument(
        "--ref_amp",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use autocast fp16 for PyTorch reference forward (CUDA only)",
    )
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


def resize_sample(rgb, depth, label, height, width):
    rgb = F.interpolate(rgb.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    depth = F.interpolate(depth.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    label = (
        F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .long()
    )
    return rgb, depth, label


def build_pytorch_model(cfg, checkpoint, device):
    model = SegModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    load_model(model, checkpoint, is_restore=False)
    model.to(device)
    model.eval()
    return model


def fast_hist(pred, label, num_classes, ignore_index):
    valid = label != ignore_index
    pred = pred[valid]
    label = label[valid]
    hist = np.bincount(num_classes * label.astype(int) + pred.astype(int), minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
    return hist


def compute_miou(hist):
    if hist.sum() == 0:
        return float("nan")
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    valid = denominator > 0
    iou = np.zeros(hist.shape[0], dtype=np.float64)
    iou[valid] = np.diag(hist)[valid] / denominator[valid]
    return float(iou[valid].mean()) if np.any(valid) else 0.0


class OnnxRunner:
    def __init__(self, onnx_path, logger):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.logger = logger
        self.logger.info(f"onnxruntime providers: {self.session.get_providers()}")

    def __call__(self, rgb, depth):
        outputs = self.session.run(
            ["logits"],
            {
                "rgb": rgb.cpu().numpy(),
                "depth": depth.cpu().numpy(),
            },
        )
        return outputs[0]


class TensorRTRunner:
    def __init__(self, engine_path):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream(device=self.device)

    def _torch_dtype(self, trt_dtype):
        mapping = {
            self.trt.float32: torch.float32,
            self.trt.float16: torch.float16,
            self.trt.int32: torch.int32,
            self.trt.int8: torch.int8,
            self.trt.bool: torch.bool,
        }
        if trt_dtype not in mapping:
            raise TypeError(f"unsupported TensorRT dtype: {trt_dtype}")
        return mapping[trt_dtype]

    def __call__(self, rgb, depth):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT validation requires CUDA")

        inputs = {"rgb": rgb.to(self.device, non_blocking=True), "depth": depth.to(self.device, non_blocking=True)}
        allocations = {}

        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))

        for name in self.tensor_names:
            shape = tuple(self.context.get_tensor_shape(name))
            torch_dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                allocations[name] = inputs[name].contiguous().to(dtype=torch_dtype)
            else:
                allocations[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.context.set_tensor_address(name, allocations[name].data_ptr())

        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        return allocations["logits"].float().cpu().numpy()


def summarize_alignment(reference_logits, other_logits):
    diff = np.abs(reference_logits - other_logits)
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_rel_diff": float(diff.mean() / (np.abs(reference_logits).mean() + 1e-6)),
    }


def main():
    args = parse_args()
    logger = get_logger()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, fallback to CPU for PyTorch reference")
        args.device = "cpu"
    device = torch.device(args.device)

    cfg = getattr(import_module(args.config), "C")
    cfg.pad = False
    dataset = build_val_dataset(cfg)
    model = build_pytorch_model(cfg, args.checkpoint, device)
    if args.ref_amp and args.device != "cuda":
        logger.warning("ref_amp requires CUDA, disabling")
        args.ref_amp = False
    logger.info(f"PyTorch reference autocast fp16: {args.ref_amp}")

    onnx_runner = OnnxRunner(args.onnx, logger) if args.onnx else None
    trt_runner = TensorRTRunner(args.engine) if args.engine else None

    hist_ref = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    hist_ref_original = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    resized_valid_pixels = 0
    original_valid_pixels = 0
    hist_onnx = np.zeros_like(hist_ref) if onnx_runner else None
    hist_trt = np.zeros_like(hist_ref) if trt_runner else None

    onnx_stats = []
    trt_stats = []
    onnx_agreement = []
    trt_agreement = []

    total = min(args.num_samples, len(dataset))
    logger.info(f"validate alignment on {total} samples, resized to {args.height}x{args.width}")

    for index in range(total):
        sample = dataset[index]
        rgb_original = sample["data"].unsqueeze(0)
        depth_original = sample["modal_x"].unsqueeze(0)
        label_original = sample["label"]

        with torch.no_grad():
            if args.ref_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ref_logits_original = model(rgb_original.to(device), depth_original.to(device)).float().cpu().numpy()
            else:
                ref_logits_original = model(rgb_original.to(device), depth_original.to(device)).float().cpu().numpy()
        ref_pred_original = ref_logits_original.argmax(axis=1)[0]
        label_np_original = label_original.cpu().numpy()
        original_valid_pixels += int((label_np_original != cfg.background).sum())
        hist_ref_original += fast_hist(ref_pred_original, label_np_original, cfg.num_classes, cfg.background)

        rgb, depth, label = resize_sample(sample["data"], sample["modal_x"], sample["label"], args.height, args.width)
        with torch.no_grad():
            if args.ref_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ref_logits = model(rgb.to(device), depth.to(device)).float().cpu().numpy()
            else:
                ref_logits = model(rgb.to(device), depth.to(device)).float().cpu().numpy()
        ref_pred = ref_logits.argmax(axis=1)[0]
        label_np = label.cpu().numpy()
        resized_valid_pixels += int((label_np != cfg.background).sum())
        hist_ref += fast_hist(ref_pred, label_np, cfg.num_classes, cfg.background)

        logger.info(f"sample {index + 1}/{total}: {sample['fn']}")

        if onnx_runner:
            ort_logits = onnx_runner(rgb, depth)
            ort_pred = ort_logits.argmax(axis=1)[0]
            onnx_stats.append(summarize_alignment(ref_logits, ort_logits))
            onnx_agreement.append(float((ort_pred == ref_pred).mean()))
            hist_onnx += fast_hist(ort_pred, label_np, cfg.num_classes, cfg.background)

        if trt_runner:
            trt_logits = trt_runner(rgb, depth)
            trt_pred = trt_logits.argmax(axis=1)[0]
            trt_stats.append(summarize_alignment(ref_logits, trt_logits))
            trt_agreement.append(float((trt_pred == ref_pred).mean()))
            hist_trt += fast_hist(trt_pred, label_np, cfg.num_classes, cfg.background)

    logger.info("========== Alignment Summary ==========")
    logger.info(
        f"PyTorch subset mIoU (original): {compute_miou(hist_ref_original):.4f} (valid_pixels={original_valid_pixels})"
    )
    logger.info(f"PyTorch subset mIoU (resized): {compute_miou(hist_ref):.4f} (valid_pixels={resized_valid_pixels})")

    if onnx_runner:
        mean_abs = np.mean([x["mean_abs_diff"] for x in onnx_stats])
        max_abs = np.max([x["max_abs_diff"] for x in onnx_stats])
        mean_rel = np.mean([x["mean_rel_diff"] for x in onnx_stats])
        logger.info(
            "ONNX vs PyTorch: mean_abs_diff=%.6f max_abs_diff=%.6f mean_rel_diff=%.6f pixel_agreement=%.6f subset_mIoU=%.4f",
            mean_abs,
            max_abs,
            mean_rel,
            float(np.mean(onnx_agreement)),
            compute_miou(hist_onnx),
        )

    if trt_runner:
        mean_abs = np.mean([x["mean_abs_diff"] for x in trt_stats])
        max_abs = np.max([x["max_abs_diff"] for x in trt_stats])
        mean_rel = np.mean([x["mean_rel_diff"] for x in trt_stats])
        logger.info(
            "TensorRT vs PyTorch: mean_abs_diff=%.6f max_abs_diff=%.6f mean_rel_diff=%.6f pixel_agreement=%.6f subset_mIoU=%.4f",
            mean_abs,
            max_abs,
            mean_rel,
            float(np.mean(trt_agreement)),
            compute_miou(hist_trt),
        )


if __name__ == "__main__":
    main()
