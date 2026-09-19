# 无人机航拍语义分割：SegFormer-B3 + 置信度过滤伪标签

基于MMSegmentation的单模型八类遥感语义分割方案。当前采用的方案为：**SegFormer-B3 + CE/Dice + 数据增强 + 0.90置信度过滤伪标签自训练 + 滑窗推理 + 12-TTA**。

本仓库由比赛工作目录导出，保留实际训练配置和源码。**主方案是conf090（A方案），不是Lovasz B方案。** 仓库中其他网络配置和历史实验脚本不代表当前主方案启用了那些模块，也不应假定每个历史实验都可直接运行。

## 当前结果

以下为项目作者提供的历史评估结果，并非此次上传时重新训练或独立复测所得。

| 方案 | 验证集mIoU（12-TTA） | 线上分数 |
| --- | ---: | ---: |
| V1：全像素硬伪标签 + CE/Dice | 77.17 | 69.42 |
| **conf090：置信度过滤伪标签 + CE/Dice** | **77.42** | **69.6674** |

验证集与线上测试集分布不同，验证指标提高不保证线上分数同步提高；本仓库不承诺重训得到完全相同的分数。

## 方法流程

1. 使用6296张人工标注训练图像训练SegFormer-B3基础教师，700张验证图像独立保留。
2. 教师为500张无标注测试图像预测类别概率。伪标签生成使用6-TTA：尺度0.75/1.0/1.25，各配原图和水平翻转。
3. 保留最大类别概率不低于0.90的像素；低于阈值的像素设为Ignore。若重新预测类别与原硬标签不同，该像素也设为Ignore，不自动改成另一个类别。本次报告中的类别分歧为0%。
4. 将6296张人工标注图像与500张过滤伪标签图像合并，每张伪样本加入一次，共6796个训练样本；学生从同一基础教师初始化重新训练。
5. 选择验证集表现最佳的学生checkpoint，执行重叠滑窗和12-TTA，导出单通道PNG提交标签。

本次500张伪标签的有效像素比例为66.16%，Ignore比例为33.84%，无全Ignore图像。0.90是模型置信度阈值，不是伪标签真实精确率保证。生成记录见[原始生成报告](outputs/pseudo_labels_v1_conf090/generation.json)。

只有在比赛规则允许使用无标注测试图像训练时，才使用测试集伪标签。验证集不参与训练。最终推理只使用一个学生模型，TTA不是多模型融合。

## 网络与训练设置

- 网络：ImageNet预训练MiT-B3主干 + SegformerHead多尺度特征融合解码头，输出8类。
- 损失：`1.0 × CrossEntropyLoss + 0.5 × DiceLoss`。
- 增强：随机缩放、90度旋转、水平翻转、光度扰动、CutOut，以及Barren优先裁剪。
- **实际训练裁剪是512×512，预处理再填充到640×640，不是真正的640×640裁剪。** 以训练pipeline内的crop_size为准。
- 学生优化器：AdamW，基础学习率2e-5，解码头学习率倍率10，AMP与梯度裁剪，1500次预热，40000次迭代，每2000次验证。
- 推理：640×640滑窗，步长480×480。
- 最终12-TTA：3个尺度0.75/1.0/1.25 × 0/180度旋转 × 原图/水平翻转；恢复坐标后对概率取平均，使用AMP，Barren系数1.0。

当前主方案没有新增CBAM、HRDA双分支、EMA、FCN辅助头或Lovasz损失。多尺度特征融合来自SegFormer本身。

## 标签约定

| 原始PNG ID | 类别 | 训练ID |
| ---: | --- | ---: |
| 0 | Ignore | 255 |
| 1 | Background | 0 |
| 2 | Building | 1 |
| 3 | Road | 2 |
| 4 | Water | 3 |
| 5 | Barren | 4 |
| 6 | Vegetation | 5 |
| 7 | Agricultural | 6 |
| 8 | Vehicle | 7 |

数据集使用`reduce_zero_label=True`。提交时将预测训练ID 0～7加1，还原为原始ID 1～8。训练用的含Ignore伪标签不能作为比赛提交结果。

## 关键文件

| 文件 | 用途 |
| --- | --- |
| [实际训练配置快照](work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py) | 当前主方案的权威配置入口 |
| [配置目录中的A配置](configs/segformer/segformer_b3_mydata_pseudo_conf090.py) | 工作区配置副本 |
| [数据集与增强定义](mmseg/datasets/voc.py) | 类别定义、旋转及类别优先裁剪 |
| [伪标签生成](demo/infer_pseudo_conf090.py) | 预测概率、过滤和来源核验 |
| [训练数据准备](tools/prepare_v1_conf090.py) | 创建独立过滤标签目录和训练配置 |
| [标签审计](tools/audit_v1_pseudo.py) | 检查原硬伪标签、样本ID和指纹 |
| [验证与概率系数评估](tools/search_barren_factor.py) | 与提交脚本相同口径的TTA评估 |
| [提交标签生成](demo/infer_gray_label.py) | 输出PNG和提交ZIP |
| [源码来源说明](SOURCE_PROVENANCE.md) | 导出范围、上游来源及发布处理 |

## 环境与复现前提

导出代码的`mmseg/version.py`标记版本为**1.2.2**，训练日志使用Python 3.8。仓库保留MMSegmentation的requirements文件，但**没有导出完整pip freeze、CUDA/cuDNN版本或容器镜像**，不能将这些requirements当作已验证的精确环境锁定文件。

优先沿用原训练环境。新环境需要安装与PyTorch/CUDA匹配的MMCV、MMEngine等依赖，参考[MMSegmentation安装说明](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)，安装兼容依赖后再执行`python -m pip install -v -e .`。

仓库不含比赛图像、人工标签PNG、过滤伪标签PNG、教师checkpoint、学生checkpoint。运行前需要自行获得使用权限并补齐：

```text
data/mydata/
  JPEGImages/                       # 训练图像及安装后的测试图像
  SegmentationClass/                # 原人工标签及原V1硬伪标签
  SegmentationClassV1Conf090/        # 真标签和过滤伪标签的独立目录
  train.txt                         # 已包含，6296行
  val.txt                           # 已包含，700行
  train_pseudo_conf090.txt           # 已包含，6796行
test_data/                          # 500张原始测试图像
checkpoints/
  teacher.pth                       # 同一基础教师，自行提供
  student_conf090.pth                # 用于评估/提交的学生，自行提供
```

若要从头重建伪标签，现有审计/准备脚本还需要原V1完整训练快照、`train_pseudo.txt`和原500张硬伪标签；这些并未全部包含在此次导出中。不得用另一份教师或标签冒充原指纹来跳过检查。已提供的generation.json是历史记录，不是500张PNG本身。

配置中保留了原服务器的绝对教师路径。下面的训练命令通过`--cfg-options`替换本机路径，不改原快照；若需要精确重现来源，还应核对generation.json中的教师SHA256。

## 训练、验证和提交

以下命令均在仓库根目录执行，前提是所需图像、标签、权重和环境已补齐。示例权重名是本地约定，不是仓库内提供的文件。

### 训练当前A方案

```bash
python tools/train.py work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py --work-dir work_dirs/conf090_retrain --cfg-options load_from=checkpoints/teacher.pth
```

使用新输出目录，不覆盖历史快照或旧权重。仍训练40000次，不启用断点resume。

### 固定12-TTA验证

```bash
python tools/search_barren_factor.py work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py checkpoints/student_conf090.pth --factors 1.0 --scales 0.75 1.0 1.25 --rotations 0 2 --hflip --amp --output-json outputs/validation_conf090_tta12.json
```

配置中原生`tools/test.py --tta`默认是6-TTA，不能将它与上述12-TTA结果混为同一设置。

### 生成提交标签

```bash
python demo/infer_gray_label.py test_data work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py checkpoints/student_conf090.pth --out-dir outputs_conf090_submission --zip-file outputs_conf090_submission/submission.zip --device cuda:0 --scales 0.75 1.0 1.25 --rotations 0 2 --hflip --amp
```

输出为1024×1024、单通道uint8 PNG，文件主名与输入一致，像素值1～8，ZIP根目录直接放PNG。验证和提交必须使用同一份学生权重及相同推理设置。

## 来源与许可

本项目建立在[OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation)之上，不将上游完整框架声明为本项目原创。保留源文件版权说明，上游Apache-2.0许可证全文见[LICENSE](LICENSE)。本仓库的代码公开不代表竞赛数据或模型权重可自由再分发。

SegFormer参考：Xie et al., [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), NeurIPS 2021。
