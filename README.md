# 无人机低空航拍语义分割：SegFormer-B3 与置信度过滤自训练

基于 MMSegmentation 的单模型八类遥感语义分割方案，面向多来源航拍图像中的类别不均衡、域偏移、目标尺度差异及伪标签噪声问题。

当前完整技术路线为：

**SegFormer-B3 → 人工标签监督训练基础教师 → 6-TTA 生成并过滤伪标签 → 人工标签与过滤伪标签联合训练学生 → 滑窗推理与 12-TTA → 八类灰度标签提交。**

当前方案标识为 `conf090`，对应伪标签置信度阈值 `0.90`。记录的验证集 **mIoU 为 77.42%**，线上成绩为 **69.6674 分**。最终推理只使用一份学生模型权重，不进行多模型融合。

本仓库由实际比赛工作目录导出。方法描述以[实际训练配置快照](work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py)、[伪标签生成报告](outputs/pseudo_labels_v1_conf090/generation.json)及对应源码为依据；性能数字来自作者保存的评估与提交记录，不是本次代码发布时重新训练或独立复测的结果。

## 目录

- [1. 任务特点与设计动机](#1-任务特点与设计动机)
- [2. 整体流程与数据划分](#2-整体流程与数据划分)
- [3. 网络结构与实际改动](#3-网络结构与实际改动)
- [4. 数据增强与监督目标](#4-数据增强与监督目标)
- [5. 置信度过滤伪标签自训练](#5-置信度过滤伪标签自训练)
- [6. 学生训练参数](#6-学生训练参数)
- [7. 滑窗推理与测试时增强](#7-滑窗推理与测试时增强)
- [8. 当前性能与结果分析](#8-当前性能与结果分析)
- [9. 代码组织与复现步骤](#9-代码组织与复现步骤)
- [10. 适用范围与复现限制](#10-适用范围与复现限制)
- [11. 来源与许可](#11-来源与许可)

## 1. 任务特点与设计动机

输入为固定尺寸 `1024×1024` 的航拍图像，输出为逐像素的八类语义标签。数据来自多个来源，地物外观、拍摄条件和类别占比存在差异。车辆等目标较小，道路需要局部边缘与较长范围的结构信息，水体、农田等类别又依赖更大范围的上下文。

本方案围绕以下问题设计训练与推理流程：

| 问题 | 采用的设计 | 目的及边界 |
| --- | --- | --- |
| 小目标与大面积地物并存 | SegFormer 四阶段特征融合、随机尺度训练、原图滑窗与多尺度推理 | 结合细节和语义；不将输入统一压缩为一张低分辨率图后直接预测 |
| 类别分布不均衡 | Barren 优先裁剪、CE 与 Dice 联合监督 | 增加弱类进入训练裁剪的机会；不保证各类获得相同监督量 |
| 多来源图像外观不同 | 光度扰动、正交旋转和翻转 | 降低对单一颜色、方向的依赖，促进域泛化 |
| 训练分布与待预测图像分布不同 | 将无标注目标图像通过离线伪标签引入学生训练 | 在规则允许时进行目标域自训练；不使用测试集真值 |
| 教师预测包含错误和不确定区域 | 最大概率阈值过滤，并核验与原硬伪标签的一致性 | 减少低置信度像素提供错误类别监督的机会 |
| 实验不易追溯、标签容易混用 | 独立标签目录、配置快照、权重与标签哈希审计 | 保留原数据和对照条件，区分生成、训练和提交阶段 |

这里的“域泛化”主要指训练增强带来的设计目标，“目标域自训练”指利用无标注目标图像的伪标签训练。当前没有额外的域判别器或特征分布对齐损失。上述机制的有效性需要结合实验观察，不能仅凭设计动机宣称每个模块都有独立增益。

## 2. 整体流程与数据划分

```text
人工标注训练集：6296 张                     独立验证集：700 张
          │                                      │
          ▼                                      │
ImageNet 预训练 MiT-B3 + 八类分割头                 │
          │ 监督训练                              │
          ▼                                      │
基础教师 checkpoint                              │
          ├───────────────┐                      │
          │               │                      │
          │        500 张无标注测试图像            │
          │               │ 教师 6-TTA 概率预测    │
          │               ▼                      │
          │        置信度 ≥ 0.90 且类别核验一致     │
          │               │                      │
          │        过滤伪标签：保留 66.16% 像素     │
          │               │                      │
          │       与 6296 张人工标签样本合并        │
          │               ▼                      │
          └──初始化──► 学生训练：6796 张，40000 iter
                          │                      │
                          └────独立验证与选权重───┘
                                  │
                                  ▼
                     单个学生 + 滑窗 + 12-TTA
                                  │
                                  ▼
                     1024×1024 灰度 PNG / ZIP
```

| 数据部分 | 数量 | 用途 |
| --- | ---: | --- |
| 人工标注训练图像 | 6296 | 教师监督训练、学生联合训练 |
| 人工标注验证图像 | 700 | 权重选择和离线评估，不加入训练列表 |
| 无标注测试图像 | 500 | 生成目标域伪标签；最终也需要预测提交结果 |
| 学生联合训练列表 | 6796 | 6296 张人工标签样本 + 500 张伪标签样本，每张只列一次 |

伪样本占训练列表约 `500 / 6796 = 7.36%`，这是图像采样占比，不是有效监督像素占比。没有通过重复三次伪样本的方式增加其采样量。

**使用测试图像生成训练伪标签属于传导式设置，只能在比赛规则明确允许时使用。** 此过程不需要测试集真值；验证集仍独立保留，不能把验证标签加入学生训练。

### 标签编码

| 原始 PNG ID | 类别 | 训练 ID |
| ---: | --- | ---: |
| 0 | Ignore，不参与类别评分 | 255 |
| 1 | Background | 0 |
| 2 | Building | 1 |
| 3 | Road | 2 |
| 4 | Water | 3 |
| 5 | Barren | 4 |
| 6 | Vegetation | 5 |
| 7 | Agricultural | 6 |
| 8 | Vehicle | 7 |

数据集配置使用 `reduce_zero_label=True`。**Background 是原始 ID 1 对应的有效语义类别，不是 Ignore。** 含原始 ID 0 的过滤伪标签仅用于训练；提交时将模型训练 ID `0～7` 加 1，还原为 `1～8`。

## 3. 网络结构与实际改动

### 3.1 当前网络是什么

当前采用标准 **SegFormer-B3**，在 MMSegmentation 中由以下组件组成：

```python
model.type = 'EncoderDecoder'
model.backbone.type = 'MixVisionTransformer'
model.decode_head.type = 'SegformerHead'
model.decode_head.num_classes = 8
```

SegFormer 采用分层 Transformer 编码器与轻量多尺度解码器。分层特征为融合浅层细节和深层语义提供基础，该设计来自原始 [SegFormer 论文](https://arxiv.org/abs/2105.15203)，不是本项目新增的网络发明。

主干为 MiT-B3，四阶段设置如下。空间比例相对于送入主干的图像张量；实际训练张量还包含后文说明的 padding。

| 阶段 | 特征空间比例 | 输出通道 | Transformer 层数 | 注意力头数 | 空间缩减比例 `sr_ratio` |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage 1 | 1/4 | 64 | 3 | 1 | 8 |
| Stage 2 | 1/8 | 128 | 4 | 2 | 4 |
| Stage 3 | 1/16 | 320 | 18 | 5 | 2 |
| Stage 4 | 1/32 | 512 | 3 | 8 | 1 |

其余关键参数包括 `mlp_ratio=4`、`drop_path_rate=0.1` 和重叠 patch embedding。主干实现见 [mit.py](mmseg/models/backbones/mit.py)。

### 3.2 解码头如何融合多尺度特征

当前启用的 [SegformerHead](mmseg/models/decode_heads/segformer_head.py) 按以下顺序处理四阶段输出：

1. 每个阶段通过 `1×1 ConvModule` 投影到 256 通道。
2. 将四路特征插值到第一阶段的空间尺寸。
3. 沿通道维拼接，得到 `4×256=1024` 通道的融合输入。
4. 用 `1×1` 融合卷积压缩为 256 通道。
5. 通过分类层输出 8 类 logits，并按训练或推理需要恢复分辨率。

当前融合是**各尺度投影、对齐、拼接再融合**，不是新增的可学习尺度权重分支。配置中解码头 `dropout_ratio=0.1`，`align_corners=False`。

### 3.3 到底修改了哪些网络和模块

项目早期使用 DeepLabV3+ / ResNet-50 路线，当前改为 SegFormer-B3。需要区分“更换基线网络”“任务适配”和“新增网络结构”：

| 层面 | 当前实际采用的改动 | 是否新增网络拓扑 |
| --- | --- | --- |
| 基线选型 | 从早期 DeepLabV3+ 路线切换为 SegFormer-B3 | 更换整体基线，不是自行发明主干 |
| 主干 | 使用 MiT-B3 配置与 ImageNet 预训练初始化 | 沿用已有主干结构 |
| 输出层 | 将解码头适配为八个评分类别 | 修改分类输出维度 |
| 多尺度表示 | 启用 SegFormer 原有四阶段特征与融合头 | 沿用原有融合结构 |
| 监督目标 | 配置 CE 1.0 + Dice 0.5 | 改变训练损失，不新增推理分支 |
| 训练样本 | 类别优先裁剪、增强、置信度过滤伪标签 | 改变数据和监督流程 |
| 测试阶段 | 滑窗、概率平均 TTA、标签编码还原 | 改变推理流程，不增加模型数量 |

**当前没有在 SegFormer 上新增 CBAM 注意力模块，也没有新增高低分辨率双分支。** 网络内的注意力来自 MiT 主干本身。当前方法的主要定制工作是八类任务适配、训练样本构建、伪标签质量控制与推理流程，不应将这些描述为一套全新 Transformer 或自研注意力网络。

教师配置文件仍保留早期命名 `deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py`。文件名不决定实际架构，应读取其中的 `model.backbone`、`model.decode_head` 和最终展开配置。

## 4. 数据增强与监督目标

### 4.1 航拍图像增强

自定义变换及数据集类别定义位于 [voc.py](mmseg/datasets/voc.py)。当前学生训练 pipeline 为：

| 顺序 | 操作 | 实际参数 | 设计目的 |
| ---: | --- | --- | --- |
| 1 | 读取图像与标签 | PNG，`reduce_zero_label=True` | 保持类别映射一致 |
| 2 | 随机缩放 | 基准 `1024×1024`，比例 `[0.5, 2.0]` | 接触不同目标尺度 |
| 3 | 随机正交旋转 | `prob=0.75`；触发时选 90/180/270° | 降低对航拍方向的依赖 |
| 4 | 类别优先裁剪 | `512×512`；目标训练 ID 4，即 Barren | 增加弱类进入裁剪的机会 |
| 5 | 水平翻转 | `prob=0.5` | 与旋转组合覆盖八种正交变换 |
| 6 | 光度扰动 | 亮度 32；对比度、饱和度 `[0.6,1.4]`；色调 18 | 模拟不同来源图像的颜色变化 |
| 7 | CutOut | `prob=0.30`；1～3 个遮挡；尺寸 32/64/96 | 减少对局部纹理的过度依赖 |
| 8 | 打包与预处理 | 标准化、图像填充、标签填充 | 组成训练 batch |

`RandomCropByClass` 使用 `target_prob=0.35`、`min_target_ratio=0.003`、`cat_max_ratio=0.80`、`max_attempts=15`。其中：

- 优先采样以当前图像存在 Barren 像素为前提，不会凭空补出目标类别。
- 目标占比按裁剪内有效像素计算，Ignore 不计入分母。
- 类别最大占比约束是尝试性约束，回退分支可能返回目标占比较高但未满足全部条件的裁剪。
- CutOut 遮挡区域的标签置为训练 Ignore `255`，不将遮挡图像强行监督为原类别。

### 4.2 512 裁剪与 640 预处理不是同一件事

本次成绩对应的真实配置是：

```text
随机缩放后的图像 → 裁剪 512×512 → 标准化并填充到 640×640 → 网络
```

`data_preprocessor.size=(640,640)` 只规定训练张量填充尺寸，**没有增加裁剪中可见的真实场景范围**。图像填充值为 0，标签填充值为 255。不能把这次实验写成“真实 640×640 大视野裁剪训练”，更不能在复现时默默将裁剪也改为 640 后仍视为相同设置。

### 4.3 CE 与 Dice 联合监督

配置的总损失为：

$$
\mathcal{L}=1.0\mathcal{L}_{CE}+0.5\mathcal{L}_{Dice}.
$$

CE 使用多类 Softmax 交叉熵，`avg_non_ignore=True`，忽略训练 ID 255。Dice 使用 `use_sigmoid=False`、`activate=True`、`naive_dice=True`、`eps=1.0`，在 Softmax 输出上计算重叠型监督项。

联合监督的动机是兼顾逐像素分类与区域重叠。当前没有设置 CE 类别权重；类别不均衡处理主要还依赖前面的类别优先裁剪，不能把该配置称为“加权交叉熵方案”。

**实现口径说明：** 本仓库的 [DiceLoss](mmseg/models/losses/dice_loss.py) 将每张图的类别维和空间维一起展平计算，并不是逐类别 Dice 等权平均。其 Ignore 处理也不等同于 CE 的显式有效像素掩码：Ignore 的 one-hot 目标为全零，但概率仍进入 Dice 分母。因此，不能宣称两个损失都严格只在有效像素集合上归一化，也不能把现有实现描述为专门的类别均衡 Dice。这里忠实保留产生当前结果的代码，未在文档更新时修改损失实现。

## 5. 置信度过滤伪标签自训练

### 5.1 基础教师与原硬伪标签

首先使用人工标签训练八类 SegFormer-B3 基础教师。教师主干采用 ImageNet 预训练的 [MiT-B3 权重](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth)，随后学习本任务类别。

教师对 500 张无标注测试图像执行 6-TTA，即：

```text
尺度 {0.75, 1.0, 1.25} × {不翻转, 水平翻转} × {0°旋转} = 6
```

每个增强分支恢复到原图坐标后，对 Softmax 类别概率做平均，再取 `argmax` 形成原硬伪标签。审计记录显示原 500 张硬伪标签全部像素为类别 ID 1～8，有效像素比例 100%，没有通过 Ignore 排除不确定像素。

### 5.2 仅过滤，不重新替换已保留的类别

当前版本使用**同一基础教师**重新计算概率，并与原硬伪标签核验。设对齐后的平均概率为 `p̄(x,c)`，原硬伪标签为 `y_ref(x)`，则：

$$
\hat c(x)=\arg\max_c\bar p(x,c),\qquad q(x)=\max_c\bar p(x,c).
$$

只有同时满足以下条件才保留像素：

1. `q(x) ≥ 0.90`；
2. 新预测的原始类别 ID `ĉ(x)+1` 与 `y_ref(x)` 相同。

否则输出原始 ID 0，在加载训练标签时映射为 Ignore 255。已保留像素的类别不会被替换成另一类别。本次重新预测与原硬标签的分歧比例为 **0.000000%**，因此实际过滤来自置信度阈值。

以下为流程伪代码，不替代可运行脚本：

```python
teacher.eval()                         # 离线固定教师
for image, reference_raw_label in unlabeled_images:
    probs = mean_aligned_probabilities(
        teacher, image,
        scales=[0.75, 1.0, 1.25], rotations=[0], hflip=True)
    train_id = argmax(probs, axis=0)
    confidence = max(probs, axis=0)
    keep = ((confidence >= 0.90)
            & (train_id + 1 == reference_raw_label))
    filtered_raw_label = where(keep, reference_raw_label, 0)
    save_png(filtered_raw_label)        # 训练用，不是提交图

student.load_state_dict(teacher_weights)
for batch in mixed_training_loader:    # 6296 真标签 + 500 过滤伪标签
    images, labels = jointly_augment_images_and_labels(batch)
    images, labels = preprocess_and_pad(images, labels)
    logits = student(images)
    loss = CE(logits, labels) + 0.5 * Dice(logits, labels)
    optimizer_step_with_amp(loss)
```

`0.90` 是模型概率阈值，不代表筛选后标签有 90% 的真实精确率。教师也可能对错误类别给出高置信度，因此该方法是减少潜在噪声监督，而不是完成伪标签真值认证。

### 5.3 本次过滤结果

500 张 `1024×1024` 伪标签共 524,288,000 个像素。报告记录：有效像素 **66.16%**，Ignore **33.84%**，全 Ignore 图像 **0 张**。

| 类别 | 原硬伪标签像素 | 过滤后保留像素 | 类内保留比例 |
| --- | ---: | ---: | ---: |
| Background | 150,669,205 | 68,566,676 | 45.51% |
| Building | 94,908,436 | 75,374,134 | 79.42% |
| Road | 46,033,559 | 34,537,575 | 75.03% |
| Water | 41,362,715 | 33,696,620 | 81.47% |
| Barren | 5,281,542 | 1,631,665 | 30.89% |
| Vegetation | 124,931,660 | 102,478,407 | 82.03% |
| Agricultural | 57,886,005 | 28,378,141 | 49.02% |
| Vehicle | 3,214,878 | 2,204,427 | 68.57% |

数据来源：[generation.json](outputs/pseudo_labels_v1_conf090/generation.json)。这里统计的是教师预测类别，不是真实测试类别分布。

统一阈值对不同类别的保留率影响明显不同，尤其 Barren 只保留 30.89%。这体现了“监督质量与覆盖率”的权衡，不能只根据总过滤比例判断少数类一定改善。当前不对人工标签做这种阈值清洗，人工标注噪声仍可能存在。

### 5.4 学生初始化与数据隔离

学生从生成伪标签的**同一基础教师 checkpoint** 初始化，而不是从上一轮已训练完成的伪标签学生继续训练。教师仅负责离线生成伪标签，不在学生训练中持续更新，也不参加最终联合推理。

数据准备脚本将人工标签副本和过滤伪标签写入独立目录 `SegmentationClassV1Conf090`，生成独立列表 `train_pseudo_conf090.txt`，保持原联合列表的 6796 行及顺序。原 `SegmentationClass`、`train.txt` 和原硬伪标签不被替换。

来源核验包括教师 SHA256、原伪标签像素指纹、过滤后 PNG 哈希、样本数量、样本 ID、训练/验证 ID 交集和全 Ignore 图像检查。ID 不相交不能证明不存在同场景或近重复图像，进一步的数据独立性审查仍有必要。

本次教师 SHA256：

```text
802a698fadb363661931c3cfd44be915287c597745cbbb234305d45cfbdb6a1d
```

原硬伪标签像素指纹：

```text
00359c4b21f2be12c76e82001452a6a634161560eb521385dc8006ed8feeec60
```

## 6. 学生训练参数

以下参数来自当前实际训练快照，不应与教师初始训练的学习率混用。

| 参数 | 当前设置 |
| --- | --- |
| 初始化 | 同一基础教师完整 checkpoint，`resume=False` |
| 人工 / 伪标签样本数 | 6296 / 500，每张列一次 |
| 实际裁剪 / 预处理填充 | `512×512` / `640×640` |
| 颜色与标准化 | BGR→RGB；mean `[123.675,116.28,103.53]`；std `[58.395,57.12,57.375]` |
| batch size / workers | 2 / 4 |
| 采样器 | `InfiniteSampler`，shuffle，`drop_last=True` |
| 优化器 | `AmpOptimWrapper` 中的 AdamW |
| 基础学习率 | `2e-5` |
| 解码头学习率倍率 | 10，对应调度前名义学习率 `2e-4` |
| AdamW 参数 | betas `(0.9,0.999)`，weight decay `0.01` |
| 混合精度 | AMP，动态 loss scale |
| 梯度裁剪 | L2 norm，`max_norm=1.0` |
| 学习率预热 | 前 1500 iter，LinearLR，`start_factor=1e-6` |
| 学习率衰减 | 1500～40000 iter，PolyLR，power 1.0，最低学习率 0 |
| 总训练长度 | 40000 iterations，不是 40000 epochs |
| 验证与保存间隔 | 每 2000 iter |
| 最佳权重指标 | 验证 mIoU，越大越好 |
| 常规 checkpoint 保留数 | `max_keep_ckpts=3` |
| 随机种子 | 3407，`deterministic=False`，`cudnn_benchmark=True` |

训练期间的自动验证使用普通验证 pipeline，最佳权重由该验证 mIoU 选择。训练完成后再运行固定设置的 12-TTA 评估，并不是每 2000 iter 都执行 12-TTA 选权重。

配置还保留了历史继承产生的顶层 `optimizer=SGD(...)` 字段；MMEngine 当前实际使用的是 `optim_wrapper.optimizer` 中的 **AdamW**。阅读参数时应以运行入口实际消费的字段为准。

## 7. 滑窗推理与测试时增强

### 7.1 重叠滑窗

验证 pipeline 不先将原始图像缩小，模型使用：

```python
test_cfg = dict(
    mode='slide',
    crop_size=(640, 640),
    stride=(480, 480))
```

每个窗口产生类别 logits，重叠像素按照窗口覆盖次数平均。边缘窗口会调整位置以覆盖图像，最后恢复整图预测。对应实现为 [EncoderDecoder](mmseg/models/segmentors/encoder_decoder.py) 的滑窗推理。

该策略在可控的单窗口计算量下处理整图；它仍是局部窗口推理，不等同于新增全局/局部双分支网络，也不能保证窗口间没有边界误差。

### 7.2 6-TTA 与 12-TTA 的区别

| 使用阶段 | 尺度 | 旋转角度 | 水平翻转 | 组合数 |
| --- | --- | --- | --- | ---: |
| 生成训练伪标签 | 0.75、1.0、1.25 | 0° | 原图与翻转 | 6 |
| 配置原生 `tools/test.py --tta` | 0.75、1.0、1.25 | 0° | 原图与翻转 | 6 |
| 当前最终验证和提交 | 0.75、1.0、1.25 | 0°、180° | 原图与翻转 | 12 |

自定义脚本中 `--rotations 0 2` 表示旋转次数分别为 0 和 2，即 0°与 180°；不是 0°与 2°。最终组合数为 `3×2×2=12`。

每个增强分支内部仍执行滑窗。分支输出逆翻转、逆旋转，在 Softmax 后将概率插值到原图尺寸，再对 12 组概率平均，最后取 `argmax`。

需要区分两层平均：**滑窗重叠区平均 logits，TTA 分支之间平均概率。** 不是对 12 张离散标签图直接投票。当前没有额外调整某一类别的概率；评估脚本中的 `--factors 1.0` 表示 Barren 概率不乘额外偏置。

TTA 复用同一份学生权重，是单模型测试时增强。它会增加计算量；本仓库没有提供统一硬件条件下的 FPS 或端到端耗时基准，因此不宣称固定倍数的实际耗时。

### 7.3 提交格式检查

[infer_gray_label.py](demo/infer_gray_label.py) 完成以下操作：

- 检查输入可读性、尺寸和输出同名冲突。
- 将训练 ID `0～7` 还原为原始 ID `1～8`。
- 保存 `1024×1024`、单通道 `uint8`、PIL `L` 模式 PNG。
- 保持输入图片的主文件名，ZIP 根目录直接放 PNG。
- 出现未成功生成的图像时停止打包，避免提交不完整结果。

可视化灰度拉伸图不是提交标签。过滤伪标签中含有 Ignore 0，也不能直接作为提交结果。

## 8. 当前性能与结果分析

### 8.1 已记录结果

验证集包含 700 张图像。`IoU=TP/(TP+FP+FN)`；mIoU 为八类 IoU 的均值，aAcc 为有效像素总体准确率；表中 Barren Acc 为该类召回率 `TP/(TP+FN)`，不是精确率。

| 方案 | 最终 TTA | 验证 mIoU (%) | aAcc (%) | Barren IoU (%) | Barren Acc (%) | 线上分数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 全像素硬伪标签 | 6 | 77.01 | 87.85 | 52.78 | 76.15 | 69.3727 |
| V1 全像素硬伪标签 | 12 | 77.17 | 87.91 | 53.19 | 76.56 | 69.42 |
| **当前 conf090 过滤伪标签** | **12** | **77.42** | **88.20** | **52.65** | **70.98** | **69.6674** |

所有行均为作者提供的历史记录。当前导出包未包含完整训练/评估日志和最终学生 checkpoint；本表不应被视为第三方已经复核的基准结果。当前 conf090 尚未提供完整八类 IoU 表，因此不从其他实验借用逐类结果补齐。

### 8.2 可以从结果得出什么

- V1 从 6-TTA 切换到 12-TTA，验证 mIoU 从 77.01% 增至 77.17%，线上记录也小幅提高。
- 相对 12-TTA 的全像素硬伪标签版本，conf090 的验证 mIoU 提高 **0.25 个百分点**，aAcc 提高 **0.29 个百分点**。
- 线上分数由 69.42 增至 69.6674，按已报告数值计算提高约 **0.2474 分**；其中对照线上成绩仅记录到两位小数。
- **提升并非所有类别同时发生。** Barren IoU 从 53.19% 降至 52.65%，下降 0.54 个百分点；该类召回率也降低。更严格过滤可能减少其监督覆盖，但仅凭这组结果不能确认单一因果。

当前结果支持“在本次实验中，过滤低置信度伪标签改善了整体指标”的观察，不支持“所有类别均提高”或“某个网络模块独立带来固定增益”的结论。没有多随机种子重复实验和置信区间，暂不作统计显著性声明。

### 8.3 为什么本地 mIoU 与线上分数不同

验证集和线上测试集不是同一批样本，可能具有不同的场景、来源和类别分布。当前又使用了目标图像伪标签训练，因此应明确区分验证集评估与线上目标集结果。

必须固定 checkpoint、滑窗尺寸和步长、TTA 组合、AMP 设置及标签映射后再比较。验证集提高不保证线上同比提高，也不能将这两个数值直接当成同一个测试集上的误差。

## 9. 代码组织与复现步骤

### 9.1 关键文件

| 文件 | 职责 |
| --- | --- |
| [实际训练配置快照](work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py) | 当前学生实验的权威入口，保留完整展开参数 |
| [配置目录副本](configs/segformer/segformer_b3_mydata_pseudo_conf090.py) | 同方案配置入口 |
| [MiT 主干](mmseg/models/backbones/mit.py) | 分层 Transformer 特征提取 |
| [SegformerHead](mmseg/models/decode_heads/segformer_head.py) | 四阶段投影、对齐、拼接和八类预测 |
| [数据集及自定义增强](mmseg/datasets/voc.py) | 八类映射、RandomRotate90、RandomCropByClass |
| [原伪标签审计](tools/audit_v1_pseudo.py) | 检查样本、标签 ID、Ignore 和像素指纹 |
| [过滤伪标签生成](demo/infer_pseudo_conf090.py) | 同教师概率重算、阈值过滤、原标签核验和报告 |
| [隔离训练数据准备](tools/prepare_v1_conf090.py) | 复制独立标签目录、保存列表与完整训练配置 |
| [训练入口](tools/train.py) | MMEngine 模型训练与周期验证 |
| [原生测试入口](tools/test.py) | 无 TTA 或配置内原生 TTA 评估 |
| [自定义 TTA 评估](tools/search_barren_factor.py) | 支持旋转 TTA；固定 factor=1.0 时不调整类别概率 |
| [提交标签生成](demo/infer_gray_label.py) | 同口径多尺度、旋转、翻转预测及 PNG/ZIP 检查 |
| [伪标签历史报告](outputs/pseudo_labels_v1_conf090/generation.json) | 教师来源、过滤统计、逐图记录与哈希 |
| [源码来源说明](SOURCE_PROVENANCE.md) | 导出范围、上游来源、发布处理和缺失材料 |

仓库保留了框架原有配置及工作目录中的辅助文件；某个文件存在不代表当前模型启用了其中功能。请以本节指定的训练快照为准。

### 9.2 环境与所需材料

源码中 `mmseg/version.py` 标记为 **1.2.2**，历史训练日志使用 Python 3.8。导出包保留 requirements，但没有完整的 `pip freeze`、PyTorch/CUDA/cuDNN 版本锁定或原容器镜像。不能把当前 requirements 当作精确复现环境锁文件。

优先沿用原训练环境；新环境应参考 [MMSegmentation 安装说明](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)，安装相互兼容的 PyTorch、MMCV 和 MMEngine，再安装本地代码：

```bash
python -m pip install -v -e .
```

本仓库**没有比赛图像、标签 PNG、教师和学生权重**。完整运行前需自行获得授权并补齐：

```text
data/mydata/
  JPEGImages/                       # 人工标签图像和安装后的 500 张测试图像
  SegmentationClass/                # 原人工标签及原 V1 硬伪标签
  SegmentationClassV1Conf090/        # 独立人工标签副本与过滤伪标签
  train.txt                         # 仓库已含：6296 行
  val.txt                           # 仓库已含：700 行
  train_pseudo.txt                  # 重建过滤实验时需另行提供原联合列表
  train_pseudo_conf090.txt           # 仓库已含：6796 行
test_data/                          # 500 张原始无标注测试图像
checkpoints/
  teacher.pth                       # 生成伪标签及初始化学生的同一教师
  student_conf090.pth                # 训练后选出的学生权重
```

`generation.json` 是历史记录，不是 500 张伪标签 PNG 本身。教师文件名中带有 `68` 也不保证重新得到的权重有某个固定线上成绩；精确追溯需比对权重内容的 SHA256。

### 9.3 准备伪标签与联合训练数据

下面以 Linux/AutoDL 仓库根目录为工作目录。涉及生成的目录应使用新的路径，已有结果不需要删除或覆盖。

先审计原硬伪标签：

```bash
python tools/audit_v1_pseudo.py --data-root data/mydata --real-list train.txt --combined-list train_pseudo.txt --seg-dir SegmentationClass
```

确认原标签与教师来源后，使用同一基础教师生成过滤伪标签。以下示例将新报告放入独立目录，不覆盖仓库携带的历史报告：

```bash
python demo/infer_pseudo_conf090.py test_data configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py checkpoints/teacher.pth --reference-dir data/mydata/SegmentationClass --out-dir outputs/pseudo_labels_conf090_rebuild --conf-thresh 0.90 --scales 0.75 1.0 1.25 --rotations 0 --hflip --amp
```

要严格复现本次实验，教师权重、教师展开配置、原硬伪标签和推理实现均需与历史报告一致。若只是以新教师运行同一方法，应保存新的报告，并将其标注为新实验，不冒用原指纹或成绩。

`tools/prepare_v1_conf090.py` 需要**原全像素硬伪标签学生训练时保存的完整配置**作为基准，以保持模型、损失、采样顺序和优化器不变。该原始完整快照需另行补齐，不能用当前过滤标签训练快照冒充。下面的 `reproduction_inputs/v1_saved_config.py` 是占位路径，不是仓库已提供文件。

```bash
python tools/prepare_v1_conf090.py reproduction_inputs/v1_saved_config.py --pseudo-dir outputs/pseudo_labels_conf090_rebuild --teacher checkpoints/teacher.pth --output-seg-dir SegmentationClassConf090Rebuild --output-list train_pseudo_conf090_rebuild.txt --output-config configs/segformer/segformer_b3_conf090_rebuild.py --work-dir work_dirs/conf090_rebuild
```

该命令使用全新的标签、列表和配置路径，遇到已有输出会拒绝覆盖；不会删除原人工标签或原硬伪标签。使用这一重建路线时，后续训练、验证和提交统一改用新生成的 `configs/segformer/segformer_b3_conf090_rebuild.py`。

### 9.4 使用已恢复的当前标签训练学生

若已补齐与当前快照一致的 `SegmentationClassV1Conf090` 和 `train_pseudo_conf090.txt`，可直接使用仓库保存的实际配置。先检查配置能否展开：

```bash
python tools/misc/print_config.py work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py
```

然后训练：

```bash
python tools/train.py work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py --work-dir work_dirs/conf090_retrain --cfg-options load_from=checkpoints/teacher.pth
```

配置保留原服务器的绝对 `load_from` 路径，因此命令显式替换为本机教师路径。此处是从教师初始化的新训练，不添加 `--resume`。输出目录使用新目录，避免覆盖历史权重。

训练后选择新目录中保存的最佳学生 checkpoint，记录其真实文件名及 SHA256；后续示例的 `checkpoints/student_conf090.pth` 代表这份权重，不应凭空假定最佳迭代数一定为 40000。

### 9.5 验证集评估

无 TTA：

```bash
python tools/test.py work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py checkpoints/student_conf090.pth
```

固定 12-TTA，所有类别使用原始概率：

```bash
python tools/search_barren_factor.py work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py checkpoints/student_conf090.pth --factors 1.0 --scales 0.75 1.0 1.25 --rotations 0 2 --hflip --amp --output-json outputs/validation_conf090_tta12.json
```

脚本读取 `val_dataloader` 对应的验证图像和真值，过滤 Ignore 后累计混淆矩阵计算指标。原生 `tools/test.py --tta` 使用配置内的 6-TTA，不要将其与上述 12-TTA 混为同一设置。

### 9.6 导出提交标签

```bash
python demo/infer_gray_label.py test_data work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py checkpoints/student_conf090.pth --out-dir outputs_conf090_submission --zip-file outputs_conf090_submission/submission.zip --device cuda:0 --scales 0.75 1.0 1.25 --rotations 0 2 --hflip --amp
```

验证和提交必须使用同一份学生权重、同一份模型配置和相同 TTA 设置。不要将含 0 的训练伪标签目录或可视化目录打包提交。

## 10. 适用范围与复现限制

- **方法定位：** 面向八类航拍语义分割的工程化适配与置信度过滤自训练；SegFormer 和 MMSegmentation 的基础网络设计属于上游工作。
- **模型数量：** 教师用于离线生成训练数据；最终只部署一个学生 checkpoint，TTA 不需要其他模型权重。
- **噪声边界：** 当前过滤的是伪标签中的低置信度区域，没有验证所有人工标注错误，也不能消除教师的高置信度错误。
- **评估边界：** 阈值、TTA 等选择依赖固定验证集，反复调参可能造成验证集选择偏差；线上提高也不等同于对其他数据集的泛化保证。
- **数值复现：** 种子固定为 3407，但未启用完全确定性计算，软硬件、数据顺序和 AMP 均可能带来差异。不承诺重新训练逐像素一致或得到完全相同分数。
- **材料完整性：** 当前发布包含源码、配置、数据划分列表和生成报告，不是包含所有数据、权重与环境的完整可执行镜像。

建议每次实验保存最终展开配置、数据列表、环境版本、教师/学生 SHA256、伪标签报告、验证 JSON、最终提交 ZIP 与线上成绩的对应关系，避免只凭文件名追溯模型。

## 11. 来源与许可

本项目基于 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation)，保留源文件版权声明。上游 Apache-2.0 许可证全文见 [LICENSE](LICENSE)，源码导出及发布范围见 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。代码公开不代表比赛图像、标签或模型权重可自由再分发。

主要参考：

- Xie et al. [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203). NeurIPS 2021.
- OpenMMLab. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)，本仓库源码版本标记为 1.2.2，未保留原工作目录的上游 git commit。
- [MMSegmentation SegformerHead 实现](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/decode_heads/segformer_head.py)，用于说明原有多尺度解码结构。
