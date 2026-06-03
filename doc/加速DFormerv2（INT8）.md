### 量化
PTQ，混合PTQ，QAT。

#### PTQ
不管是NCNN，MNN，还是TensorRT，都提供相关工具进行PTQ。以TensorRT为例，首先准备校准数据，然后使用相关工具，trtexec 的命令
。转换好后，验证100张数据，如下结果：
```bash
========== Alignment Summary ==========
int 8 vs PyTorch: mean_abs_diff=0.601687 max_abs_diff=12.957222 mean_rel_diff=0.182346 pixel_agreement=0.880279 subset_mIoU=0.4993
```
量化后的相似度只有0.878380，对结构（注意力 + LayerNorm + Softmax），INT8 掉点大很常见。

#### 混合精度 INT8
还是PTQ，只是在量化时，使用混合精度。让网络主体（Conv/Linear/MatMul）INT8，但对敏感算子保 FP16（常见敏感项：Softmax、LayerNorm、部分归一化/激活）。

trtexec 做法是同时开 --int8 和 --fp16。TensorRT 会在可量化的层用 INT8，不适合 INT8 的层可能会保留 FP16/FP32（取决于算子与硬件支持），这就是最常见的混合精度。trtexec 基本做不到按层精确指定精度这种细粒度混合策略。
```bash
========== Alignment Summary ==========
f16mix vs PyTorch: mean_abs_diff=0.609373 max_abs_diff=13.249213 mean_rel_diff=0.184646 pixel_agreement=0.878380 subset_mIoU=0.5022
```
混合精度 INT8，subset_mIoU 明显更高。

#### QAT
QAT：用蒸馏结果 epoch-68_miou_52.05.pth，对 Conv/Linear 做 fake quant，低学习率微调 20~40 epoch，再重新导出 TensorRT INT8。