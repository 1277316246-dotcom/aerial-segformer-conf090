# 源码来源与发布范围

## 来源

- 输入：作者提供的 `segformer_b3_conf090_code_20260919_135302.tar.gz`，由AutoDL训练项目导出。
- 输入压缩包SHA256：`58ecba336a8318bb883cc2d49729526f2976bfd4c0e2a7ec6ec2406121a353b6`。
- 主方案：conf090 / A，线上69.6674分为作者提供的历史结果。
- 框架：OpenMMLab MMSegmentation；导出源码中版本标记为1.2.2。没有保留原git commit，因此不能声称整个导出目录与官方v1.2.2标签完全一致。
- 官方来源：https://github.com/open-mmlab/mmsegmentation
- LICENSE原文对应：https://raw.githubusercontent.com/open-mmlab/mmsegmentation/v1.2.2/LICENSE

## 项目定制内容

主要包括数据集八类映射、RandomRotate90、RandomCropByClass、个人数据集配置、SegFormer-B3训练配置、伪标签生成及过滤、标签审计、验证和提交脚本。原目录还保留了一些历史实验或辅助模块；其存在不代表它们被当前A方案启用。

复现入口以 `work_dirs/segformer_b3_mydata_pseudo_conf090/segformer_b3_mydata_pseudo_conf090.py` 为准。不要将其他实验配置或同名后来修改过的配置当作该次训练设置。

## 发布时的处理

- 从原压缩包解压保留源码、配置、划分列表和generation.json。
- 不发布Jupyter自动备份目录、Python缓存或自动生成的`mmseg/.mim`链接。这些链接可在安装时重建；其中model-index/dataset-index链接在原压缩包中没有对应的根目录目标文件。
- 新增README、Git忽略规则、换行规则及本来源说明，补回上游许可证全文。
- 不修改导出的训练/推理逻辑、实际A配置、划分列表或原生成报告。发布前逐文件与原压缩包比对保留文件。
- 原压缩包没有模型权重；比赛图像和标签PNG未打包。仓库可能保留上游demo示例图片，它们不是比赛训练/测试数据。
- 发布前做了常见凭据格式的启发式扫描；没有命中不等同于完备安全审计。

## 复现限制

本次导出不是包含全部数据和环境的完整可执行实验镜像。教师/学生权重、真实图像、标签、过滤伪标签PNG、完整环境锁定信息，以及原V1的部分来源文件需另行补充。generation.json保存路径和哈希用于追溯，不提供这些文件本身。

公开源代码不构成对未随仓库分发的数据或权重的授权。保留各源文件已有版权和许可声明；LICENSE为上游框架许可证。
