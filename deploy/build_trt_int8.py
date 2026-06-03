import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine from ONNX with calibrator")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine path")
    parser.add_argument("--calib_dir", required=True, help="Calibration directory with *.npz and index.txt")
    parser.add_argument("--calib_cache", default=None, help="Calibration cache path (default: engine + .calib)")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workspace_mb", type=int, default=4096)
    parser.add_argument(
        "--fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FP16 kernels where possible (recommended for Jetson)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@dataclass(frozen=True)
class CalibSample:
    rgb: np.ndarray
    depth: np.ndarray


def load_index(calib_dir):
    index_path = os.path.join(calib_dir, "index.txt")
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    with open(index_path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]
    if not paths:
        raise RuntimeError(f"empty index file: {index_path}")
    return paths


def parse_onnx_model(parser, onnx_path):
    onnx_path = os.path.abspath(onnx_path)

    # Prefer file-based parsing so TensorRT can resolve external weight files
    # such as "<model>.onnx.data" relative to the ONNX file location.
    if hasattr(parser, "parse_from_file"):
        if parser.parse_from_file(onnx_path):
            return
    else:
        onnx_dir = os.path.dirname(onnx_path) or "."
        prev_cwd = os.getcwd()
        try:
            os.chdir(onnx_dir)
            with open(onnx_path, "rb") as f:
                if parser.parse(f.read()):
                    return
        finally:
            os.chdir(prev_cwd)

    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    onnx_data_path = onnx_path + ".data"
    if os.path.exists(onnx_data_path) and "Failed to open file" in errors:
        errors += (
            "\nHint: this ONNX uses external data. Parse it from the file path "
            "instead of in-memory bytes so TensorRT can resolve the .onnx.data file."
        )
    raise RuntimeError("ONNX parse failed:\n" + errors)


class NpzCalibrator:
    def __init__(self, trt, paths, batch_size, height, width, cache_path, seed):
        self.trt = trt
        self.paths = list(paths)
        self.batch_size = int(batch_size)
        self.height = int(height)
        self.width = int(width)
        self.cache_path = cache_path
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.paths)

        if not torch.cuda.is_available():
            raise RuntimeError("INT8 calibration requires CUDA")
        self.device = torch.device("cuda")
        self._current = 0
        self._batch_tensors = {}

    def get_batch_size(self):
        return self.batch_size

    def _read_npz(self, path):
        data = np.load(path)
        rgb = data["rgb"].astype(np.float32, copy=False)
        depth = data["depth"].astype(np.float32, copy=False)
        if rgb.shape != (3, self.height, self.width):
            raise ValueError(f"rgb shape mismatch for {path}: {rgb.shape} != (3,{self.height},{self.width})")
        if depth.shape != (3, self.height, self.width):
            raise ValueError(f"depth shape mismatch for {path}: {depth.shape} != (3,{self.height},{self.width})")
        return CalibSample(rgb=rgb, depth=depth)

    def get_batch(self, names):
        if self._current >= len(self.paths):
            return None

        end = min(self._current + self.batch_size, len(self.paths))
        batch_paths = self.paths[self._current : end]
        self._current = end

        rgbs = []
        depths = []
        for p in batch_paths:
            s = self._read_npz(p)
            rgbs.append(s.rgb)
            depths.append(s.depth)

        rgb_np = np.stack(rgbs, axis=0)
        depth_np = np.stack(depths, axis=0)

        rgb = torch.from_numpy(rgb_np).to(self.device, non_blocking=True)
        depth = torch.from_numpy(depth_np).to(self.device, non_blocking=True)

        self._batch_tensors = {"rgb": rgb, "depth": depth}
        return [int(self._batch_tensors[n].contiguous().data_ptr()) for n in names]

    def read_calibration_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return None
        with open(self.cache_path, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def main():
    args = parse_args()
    if not os.path.exists(args.onnx):
        raise FileNotFoundError(args.onnx)

    if args.calib_cache is None:
        args.calib_cache = args.engine + ".calib"

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    parse_onnx_model(parser, args.onnx)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_mb) * 1024 * 1024)

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    profile.set_shape("rgb", (args.batch_size, 3, args.height, args.width), (args.batch_size, 3, args.height, args.width), (args.batch_size, 3, args.height, args.width))
    profile.set_shape("depth", (args.batch_size, 3, args.height, args.width), (args.batch_size, 3, args.height, args.width), (args.batch_size, 3, args.height, args.width))
    config.add_optimization_profile(profile)

    calib_paths = load_index(args.calib_dir)

    class TRTCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, impl):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.impl = impl

        def get_batch_size(self):
            return self.impl.get_batch_size()

        def get_batch(self, names):
            return self.impl.get_batch(names)

        def read_calibration_cache(self):
            return self.impl.read_calibration_cache()

        def write_calibration_cache(self, cache):
            return self.impl.write_calibration_cache(cache)

    impl = NpzCalibrator(trt, calib_paths, args.batch_size, args.height, args.width, args.calib_cache, args.seed)
    config.int8_calibrator = TRTCalibrator(impl)

    if hasattr(builder, "build_serialized_network"):
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("build_serialized_network returned None")
        os.makedirs(os.path.dirname(args.engine), exist_ok=True)
        with open(args.engine, "wb") as f:
            f.write(engine_bytes)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("build_engine returned None")
        os.makedirs(os.path.dirname(args.engine), exist_ok=True)
        with open(args.engine, "wb") as f:
            f.write(engine.serialize())

    print("onnx:", args.onnx)
    print("engine:", args.engine)
    print("int8_calib_cache:", args.calib_cache)
    print("calib_samples:", len(calib_paths))
    print("shape:", f"batch={args.batch_size}, rgb/depth=3x{args.height}x{args.width}")
    print("fp16:", bool(args.fp16))


if __name__ == "__main__":
    main()
