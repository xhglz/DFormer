### 部署
从蒸馏结果导出 ONNX 模型，再用 TensorRT 构建 FP16 engine。

#### onnx

**两个输入：**
- rgb
- depth

**一个输出：**
- logits

**固定输入尺寸：**
- 1x3x384x512

#### tensorrt
用 trtexec 从 ONNX 构建固定 shape 的 TensorRT FP16 engine。验证 engine 是否正常运行。
```bash
trtexec --loadEngine=deploy/engine/dformerv2_s_lite_mlp_fp16.engine --shapes=rgb:1x3x384x512,depth:1x3x384x512 --fp16
```


#### 验证
在验证集（100张）上，验证 TensorRT 输出和 PyTorch/ONNX 是否一致。

1. 结果对齐
PyTorch 和 ONNX 对比，PyTorch 和 TensorRT 对比。以前很麻烦，现在可以用AI了。
```text
========== Alignment Summary ==========
PyTorch subset mIoU (original): 0.5247 (valid_pixels=25687342)
PyTorch subset mIoU (resized): 0.4763 (valid_pixels=16433762)
ONNX vs PyTorch: mean_abs_diff=0.000712 max_abs_diff=0.014459 mean_rel_diff=0.000216 pixel_agreement=0.999884 subset_mIoU=0.4763
TensorRT vs PyTorch: mean_abs_diff=0.131576 max_abs_diff=4.495104 mean_rel_diff=0.039856 pixel_agreement=0.973562 subset_mIoU=0.4771
```
- PyTorch ： original 0.5247 、 resized 0.4763 ，并且 valid_pixels 规模正常，说明数据/label/ignore 处理正确，resize 到 384x512 确实会掉点（预期现象）。
- ONNX 对齐通过 ： pixel_agreement=0.999884 、 subset_mIoU=0.4763 与 PyTorch resized 0.4763 一致，且 mean_abs_diff 很小，说明 ONNX 导出与 ONNXRuntime 推理基本等价于 PyTorch reference 。
- TensorRT 基本可用但有 FP16 漂移 ：
  - pixel_agreement=0.973562 ，说明约 2.6% 像素的 argmax 跟 PyTorch 不同。
  - subset_mIoU=0.4771 与 PyTorch 0.4763 接近（甚至略高一点，属于样本波动），说明整体语义分割结果并没有被破坏。
  - 但 mean_abs_diff=0.131576 、 max_abs_diff=4.495104 说明 logits 数值有少量明显偏差（典型 FP16/融合算子造成）。


2. 性能测试
GPU/CPU/Latency/内存/温度/功耗。 本次只看计算Latency，考虑部署时会把 logits/mask 拷回 CPU 做后处理。
```
engine: deploy/engine/dformerv2_s_lite_mlp_fp16.engine
shape: rgb/depth = 1x3x384x512
warmup: 50 iters: 200 copy_output: True
gpu latency(ms): mean=1.945 p50=1.947 p90=1.956 p95=1.958 p99=1.963 min=1.915 max=1.969 fps=514.17
e2e latency(ms): mean=5.469 p50=5.258 p90=5.316 p95=5.358 p99=15.312 min=5.208 max=15.323 fps=182.84
wall_fps: 182.80
```
