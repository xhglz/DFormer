**DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation**(cvpr2025) 

### 软件环境
50系显卡是sm120, 需要提高pytorch版本。安装依赖如下
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

MMCV_WITH_OPS=1 pip install mmcv==2.2.0 

pip install timm
```

### 目标
测试数据 NYUDepthv2，测试模型DFormerv2_S。
- 使用DFormerv2_S建立baseline，评估其在不同指标下的性能。
- 测量miou、FPS、latency、params、FLOPs和显存等关键指标。

#### 测试指令
测试脚本
```bash
# 获取 miou macc mf1 ious
bash eval.sh
# 获取 FLOPs Parameters
PYTHONPATH=. python utils/benchmark.py --config local_configs.NYUDepthv2.DFormerv2_S
# 获取 Latency
PYTHONPATH=. python utils/latency.py --config local_configs.NYUDepthv2.DFormerv2_S
```
miou macc mf1 ious 结果
```bash
eval.py[line:205] - INFO: miou:56.2, macc:70.07, mf1:69.99
eval.py[line:206] - INFO: ious:[83.1500015258789, 89.33999633789062, 66.37999725341797, 74.86000061035156, 69.37999725341797, 65.52999877929688, 55.959999084472656, 47.02000045776367, 49.77000045776367, 49.130001068115234, 68.91999816894531, 72.5199966430664, 61.540000915527344, 31.229999542236328, 23.25, 69.47000122070312, 64.08999633789062, 49.93000030517578, 57.61000061035156, 41.060001373291016, 26.969999313354492, 73.02999877929688, 38.15999984741211, 71.13999938964844, 70.33000183105469, 40.150001525878906, 50.970001220703125, 43.9900016784668, 18.8700008392334, 87.05000305175781, 88.12999725341797, 55.599998474121094, 79.80999755859375, 65.48999786376953, 53.66999816894531, 62.689998626708984, 25.389999389648438, 37.88999938964844, 24.709999084472656, 43.939998626708984]
```
FLOPs Parameters 结果
```bash
the flops is 33.85G,the params is 26.69M
```
Latency 结果
```bash
Avg Latency=27.421670665740965ms
```

#### 指标结果
| 指标 | miou | macc | mf1 | FLOPs | params | Latency |
|---|---|---|---|---|---|---|
| 数值 | 56.2 | 70.07 | 69.99 | 33.85G | 26.69M | 27.42ms |


#### 指标含义
1. **mIoU** (mean Intersection over Union)，对每个类别计算预测区域与真实区域的交并比，再取平均。
$$\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|} = \frac{TP_c}{TP_c + FP_c + FN_c} $$

$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}_c$$

其中 $P_c$ 为预测为第 $c$ 类的像素集合，$G_c$ 为真实标签为第 $c$ 类的像素集合。$TP_c, FP_c, FN_c$ 分别为真正例、假正例、假负例。**结果 56.2% mIoU 说明 DFormer 在 NYUv2 上表现不错**（当前 SOTA 类方法在该数据集通常 50-58%）。

2. **mAcc** (mean Accuracy)，每个类别的分类准确率的均值。
$$\text{Acc}_c = \frac{TP_c}{TP_c + FN_c}$$

$$\text{mAcc} = \frac{1}{C} \sum_{c=1}^{C} \text{Acc}_c$$

3. **mF1** (mean F1 Score)，每个类别的 F1 分数的均值，F1 是精确率（Precision）和召回率（Recall）的均衡指标。mF1(69.99) 与 mAcc(70.07) 非常接近，说明整体上 Precision ≈ Recall，模型预测的精确率和召回率比较均衡。
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{mF1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c$$

4. **FLOPs**(Floating Point Operations)，模型单次前向推理所需的浮点运算次数（乘加运算）。衡量模型的计算复杂度，单位：GFLOPs = (10^9) 次浮点运算。例如：对于卷积层（输入 $C_{in} \times H \times W$，卷积核 $K \times K$，输出 $C_{out}$，输出尺寸 $H' \times W'$）：

$$\text{FLOPs}{\text{conv}} = 2 \cdot K^2 \cdot C{in} \cdot C_{out} \cdot H' \cdot W'$$

5. **Params** (Parameters)，模型所有可训练参数的总数。衡量模型的存储空间大小。单位：M（百万），例如 26.69M ≈ $2.669 \times 10^7$ 个参数，每个 float32 参数占用 4 字节，因此模型大小 ≈ $26.69 \times 4 \approx 106.8$ MB。
$$\text{Params} = \sum_{\text{layer}} \text{layer 参数量}$$


6. **Latency**，模型单次前向推理的单张图片耗时。单位：ms。