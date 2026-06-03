### 蒸馏提点
整体思路，先易后难。
- 先蒸馏 DFormerv2_S 到 DFormerv2_S_Lite_mlp（不需要额外 1x1 对齐层），再蒸馏 DFormerv2_S_Lite_B 到 DFormerv2_S_Lite_mlp。
- 如果后面 teacher 换成 DFormerv2_B ，再补 feature adapter。
- 先不全量蒸馏。3项：分割监督 CE loss，logit 蒸馏（KL），feature 蒸馏（对齐 stage3 和 stage4 特征）。
- 损失函数：L = L_ce + 0.5 * L_kd + 0.1 * L_feat。L_ce : 常规 segmentation cross entropy，L_kd : student logits 和 teacher logits 的 KL distillation，L_feat : stage3 + stage4 的 feature L2 loss。

#### 文件改动
蒸馏算是“微调增强”。使用AI生成的代码，然后手动调整。
- distiller.py ：蒸馏包装模块，封装 teacher = DFormerv2_S， student = DFormerv2_S_Lite_mlp 的前向与损失计算。
- train_distill.py ：蒸馏训练入口，复用现有 dataloader，eval，checkpoint 流程。
- DFormerv2_S_Lite_mlp_distill.py ：蒸馏配置，使用 student checkpoint 继续训练。
- train_distill.sh ：训练脚本，调用 train_distill.py 训练。

| Teacher  | Student Init | kd_weight | feat_weight | T | LR | Epochs | Batch | Best mIoU | 备注 |
|----------|--------------|-----------|-------------|---|----|--------|-------|-----------|-----------------|
| DFormerv2_S | S_Lite_mlp_miou_51.7.pth | 0.5 | 0.1 | 4 | 3e-5 | 100 | 8 | 52.04 | baseline distill |
| DFormerv2_S | S_Lite_mlp_miou_51.7.pth | 0.7 | 0.1 | 4 | 3e-5 | 100 | 8 | 51.9 | 提高 kd 权重 |  
| DFormerv2_S | S_Lite_mlp_miou_51.7.pth | 0.5 | 0.2 | 4 | 3e-5 | 100 | 8 | 52.05 | 提高 feat 权重 |
| DFormerv2_S | S_Lite_mlp_miou_51.7.pth | 0.5 | 0.1 | 2 | 3e-5 | 100 | 8 | 51.82 | 降低温度 |

- 如果 kd_loss 一直远大于 seg_loss ，通常是 kd_weight 太大或 T 不合适。
- 如果 kd_loss 一直几乎为 0，可能 teacher 太弱/太接近 student，或 T 太高导致信息变淡。

从结果看
- baseline distill: 52.04 - 51.70 = +0.34
- kd_weight = 0.7 : 51.90 - 51.70 = +0.20
- feat_weight = 0.2 : 52.05 - 51.70 = +0.35
- T = 2 : 51.82 - 51.70 = +0.12
如果 miou 增量 > 0.2+ 说明蒸馏有效，可以继续微调。本次不需继续了。


#### LOSS 函数
1) 语义分割交叉熵 $L_{ce}$
设：
- 输入图像大小为 $H\times W$，类别数为 $C$
- student 输出 logits 为 $z_s(p)\in\mathbb{R}^C$，其中像素 $p$ 表示 $(h,w)$
- 标签 $y(p)\in\{0,\dots,C-1\}$，忽略标签为 $255$
- 有效像素集合 $\Omega=\{p\mid y(p)\neq 255\}$

则像素级交叉熵：
$$
L_{ce}
= \frac{1}{|\Omega|}
\sum_{p\in\Omega}
\left(
-\log
\frac{\exp(z_s^{\,y(p)}(p))}
{\sum_{c=1}^{C}\exp(z_s^{\,c}(p))}
\right)
$$

2) Logits 蒸馏 KL 损失 $L_{kd}$

设：
- teacher logits 为 $z_t(p)\in\mathbb{R}^C$
- 温度为 $T>0$
- 软化后的概率分布：
$$
p_s^T(p)=\text{softmax}(z_s(p)/T),\quad
p_t^T(p)=\text{softmax}(z_t(p)/T)
$$

像素级 KL 散度（teacher 作为 target）：
$$
KL\big(p_t^T(p)\,\|\,p_s^T(p)\big)
=
\sum_{c=1}^{C}
p_t^T(p,c)\,
\log\frac{p_t^T(p,c)}{p_s^T(p,c)}
$$

对有效像素取平均，并乘上常用的 $T^2$ 缩放（与你实现一致）：
$$
L_{kd}
=
\frac{T^2}{|\Omega|}
\sum_{p\in\Omega}
KL\big(p_t^T(p)\,\|\,p_s^T(p)\big)
$$

3) 特征蒸馏 $L_{feat}$（stage3 + stage4 的 L2/MSE）

设：
- 选取的特征层集合 $\mathcal{I}=\{3,4\}$（你实现里用下标 `(2,3)`，对应第 3/4 个输出 stage）
- 第 $i$ 个 stage 的特征图：  
  $F_s^{(i)}\in\mathbb{R}^{B\times C_i\times H_i\times W_i}$，  
  $F_t^{(i)}\in\mathbb{R}^{B\times C_i\times H_i\times W_i}$  
  （你这里 teacher= DFormerv2_S，student= S_Lite，通道一致所以可直接对齐）

则每个 stage 的 MSE：
$$
\text{MSE}\big(F_s^{(i)},F_t^{(i)}\big)
=
\frac{1}{B\,C_i\,H_i\,W_i}
\left\|F_s^{(i)}-F_t^{(i)}\right\|_2^2
$$

取选定 stage 的平均：
$$
L_{feat}
=
\frac{1}{|\mathcal{I}|}
\sum_{i\in\mathcal{I}}
\text{MSE}\big(F_s^{(i)},F_t^{(i)}\big)
$$

总损失
$$
L = L_{ce} + 0.5\,L_{kd} + 0.1\,L_{feat}
$$