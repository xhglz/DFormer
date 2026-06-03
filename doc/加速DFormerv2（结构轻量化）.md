### 整体压缩流程
1. 硬件链路是否可部署。
2. 结构压缩 + 蒸馏。
3. 量化 + 板端加速。
4. 稳定性和工程封装。

### 确立压缩路线。
边缘端芯片类型： Jetson Orin/Xavier 、 RK3588 、 地平线 、 寒武纪 、 CPU。部署和硬件平台紧密相关，因此首先判断从论文到工程化的链路是否可行。

以 3588 为例，结构需要改为对 NPU 友好，而 NPU 对 transformer 风格图的支持和优化通常不如对卷积图友好（不能顺利变成 RKNN 可行的算子图）。改成 Jetson 部署，路线会比 RK3588 NPU 宽松很多，因为 Jetson 对 Transformer/MatMul/LayerNorm 这类算子更友好， DFormerv2 保留原始设计的可行性更高。

确定Jetson版的压缩路线，上一章已经确定使用DFormerv2_S的baseline，因此主线是： 
```
DFormerv2_S -> 轻量 decoder -> 蒸馏 -> ONNX -> TensorRT FP16 -> 再视情况做 INT8
```
### 结构轻量化
1. 新建配置文件 local_configs/NYUDepthv2/DFormerv2_S_Lite.py
```python
from .._base_.datasets.NYUDepthv2 import *

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormerv2_S_Lite"
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
C.decoder = "MLPDecoder"
C.decoder_embed_dim = 256
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 16
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.75, 1.0, 1.25]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.15
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [384, 512]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_MLP")
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

if not os.path.exists(config.log_dir):
  os.makedirs(config.log_dir, exist_ok=True)

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
```
- decoder="MLPDecoder" ：比默认 ham（用矩阵分解增强多尺度特征融合） 更适合部署。
- decoder_embed_dim=256。
- train_scale_array=[0.75,1.0,1.25] ：保留一点尺度鲁棒性，但不过度增加训练扰动。
- drop_path_rate=0.15。

2. 在 models/encoders/DFormerv2.py 里新增一个构造函数。
```python
def DFormerv2_S_lite(pretrained=False, **kwargs):
  model = dformerv2(
    embed_dims=[64, 128, 256, 512],
    depths=[2, 4, 12, 3],
    num_heads=[4, 4, 8, 16],
    heads_ranges=[4, 4, 6, 6],
    mlp_ratios=[4, 4, 3, 3],
    **kwargs,
  )
  return model
```
- 宽度不变： [64,128,256,512]
- 只压深度：从 S 的 [3,4,18,4] 改成 [2,4,12,3]

3. 在 models/builder.py 注册，否则 C.backbone = "DFormerv2_S_Lite" 不会被识别。
```python
if cfg.backbone == "DFormerv2_S_Lite":
  from .encoders.DFormerv2 import DFormerv2_S_lite as backbone

  self.channels = [64, 128, 256, 512]
```
### 实验与结果
新建 checkpoints/NYUDepthv2/DFormerv2_S_Lite_mlp.py 和 local_configs/NYUDepthv2/DFormerv2_S_Lite_ham.py。一个使用默认的 “ham decoder”，一个使用 “MLPDecoder”，其它参数相同。受限训练算力的限制，batch_size=8，use amp，“ham” best epoch为335，“mlp” best epoch为230。都有继续训练的潜力，估计miou还会提升0.1~0.2。

| Decoder | ham | mlp | 
|---|---|---|
| mlp | 52.26 | 51.7 |

DFormerv2_S作为baseline，其他模型基于其结构压缩。
| model | flops | params | latency |
|---|---|---|---|
| base | 33.85G | 26.69M | 27.42ms |
| ham  | 26.49G | 19.78M | 21.00ms |
| mlp | 26.9G  | 19.17M | 20.18ms |




























- 训练总步数近似是： total_iteration = nepochs * niters_per_epoch
- 而 niters_per_epoch = num_train_imgs // batch_size + 1 。
- 所以 batch 减半后， niters_per_epoch 近似翻倍。







- warm_up_epoch ：
- 现在还是 10 ，但 batch 变小后，实际 warmup iteration 也变多。
- 建议从 10 改成 5
- 原因是 niters_per_epoch 已经变大， 5 epoch 的 warmup 实际步数就不短了。

lr ：
- 第一轮可以先保持 6e-5
- 如果训练明显不稳定、loss 抖动偏大，再降到 4e-5
- 不建议因为 batch 变小就立刻大改 LR