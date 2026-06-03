import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engine latency (fixed shape, batch=1)")
    parser.add_argument("--engine", required=True, help="TensorRT engine path")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument(
        "--copy_output",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Include D2H copy of logits in timing",
    )
    return parser.parse_args()


def percentile(values_ms, p):
    return float(np.percentile(values_ms, p))


@dataclass(frozen=True)
class BenchResult:
    n: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    fps: float


def summarize(latencies_ms):
    lat = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(lat.mean())
    p50_ms = percentile(lat, 50)
    p90_ms = percentile(lat, 90)
    p95_ms = percentile(lat, 95)
    p99_ms = percentile(lat, 99)
    min_ms = float(lat.min())
    max_ms = float(lat.max())
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return BenchResult(
        n=int(lat.shape[0]),
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
    )


def format_result(label, result: BenchResult):
    return (
        f"{label} latency(ms): mean={result.mean_ms:.3f} p50={result.p50_ms:.3f} p90={result.p90_ms:.3f} "
        f"p95={result.p95_ms:.3f} p99={result.p99_ms:.3f} min={result.min_ms:.3f} max={result.max_ms:.3f} "
        f"fps={result.fps:.2f}"
    )



def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT benchmark")

    import tensorrt as trt

    if not os.path.exists(args.engine):
        raise FileNotFoundError(args.engine)

    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize engine: {args.engine}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create execution context")

    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    output_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    if "rgb" not in input_names or "depth" not in input_names:
        raise RuntimeError(f"engine inputs must include rgb/depth, got inputs={input_names}")
    if "logits" not in output_names:
        raise RuntimeError(f"engine outputs must include logits, got outputs={output_names}")

    context.set_input_shape("rgb", (1, 3, args.height, args.width))
    context.set_input_shape("depth", (1, 3, args.height, args.width))

    def torch_dtype(trt_dtype):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if trt_dtype not in mapping:
            raise TypeError(f"unsupported TensorRT dtype: {trt_dtype}")
        return mapping[trt_dtype]

    allocations = {}
    for name in tensor_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = torch_dtype(engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name in ("rgb", "depth"):
                allocations[name] = torch.randn(shape, dtype=dtype, device=device)
            else:
                allocations[name] = torch.zeros(shape, dtype=dtype, device=device)
        else:
            allocations[name] = torch.empty(shape, dtype=dtype, device=device)
        context.set_tensor_address(name, allocations[name].data_ptr())

    start_evt = torch.cuda.Event(enable_timing=True)
    mid_evt = torch.cuda.Event(enable_timing=True)

    def run_once():
        with torch.cuda.stream(stream):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 failed")
        if args.copy_output:
            stream.synchronize()
            _ = allocations["logits"].float().cpu().numpy()
        else:
            stream.synchronize()

    for _ in range(args.warmup):
        run_once()

    gpu_latencies_ms = []
    e2e_latencies_ms = []
    t0 = time.time()
    for _ in range(args.iters):
        e2e_start = time.perf_counter()
        with torch.cuda.stream(stream):
            start_evt.record(stream)
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 failed")
            mid_evt.record(stream)
        stream.synchronize()
        gpu_elapsed_ms = float(start_evt.elapsed_time(mid_evt))

        if args.copy_output:
            _ = allocations["logits"].float().cpu().numpy()
            torch.cuda.synchronize()

        e2e_end = time.perf_counter()
        e2e_elapsed_ms = (e2e_end - e2e_start) * 1000.0

        gpu_latencies_ms.append(gpu_elapsed_ms)
        e2e_latencies_ms.append(e2e_elapsed_ms)
    t1 = time.time()

    gpu_result = summarize(gpu_latencies_ms)
    e2e_result = summarize(e2e_latencies_ms)
    wall_fps = args.iters / max(t1 - t0, 1e-9)

    print("engine:", args.engine)
    print("shape: rgb/depth = 1x3x%dx%d" % (args.height, args.width))
    print("warmup:", args.warmup, "iters:", args.iters, "copy_output:", args.copy_output)
    print(format_result("gpu", gpu_result))
    print(format_result("e2e", e2e_result))
    print("wall_fps: %.2f" % wall_fps)


if __name__ == "__main__":
    main()
