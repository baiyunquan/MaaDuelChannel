# 争锋频道绿藤城训练流水线实施计划

## 目标与边界

- 仓库位于 `C:\Users\liaic\Source\Repos\MaaDuelChannel`，本阶段不修改 MAA。
- 只处理 `G:\MAA-DuelChannel\training-data` 中的绿藤城视频；其余场地仅进入扫描报告。
- 视频只读；所有缓存、标注、模型和报告写入外部 workspace。
- 交付可运行代码和小规模冒烟验证，不执行全量训练，不交付正式权重。
- CannotMax 仅作流程参考，不复制未授权代码和素材。

## 流水线

1. 建立 78 类敌人目录及可追溯素材缓存。
2. 扫描视频并按准备、归零窗口、ROUND、战斗和结束状态切分多局对决。
3. YOLO 分类准备区头像，OCR 识别数量；YOLO 检测初始站位并与清单核对。
4. 通过橙色和蓝色血条的连续帧状态生成左右胜负标签。
5. 用本地审核界面修正清单、框、站位和胜负；人工结果持久化且优先。
6. 将审核通过的每个单位编码为带类型和归一化位置的 token，训练左右交换对称的 Transformer。
7. 所有有效样本用于训练，不划分验证集或测试集。

## CLI

- `duel assets sync`
- `duel scan`
- `duel synth`
- `duel train-vision roster|battlefield --all`
- `duel extract`
- `duel review`
- `duel build-dataset`
- `duel train-predictor --all`
- `duel report`

## 验收

- 两种现有分辨率可映射到统一坐标。
- 多局视频切分、OCR、清单核对、血条胜负和人工修正均有自动化测试。
- 合成夹具可完成扫描、抽取、数据集构建及一轮训练冒烟。
- 输入目录不被修改，重复运行可恢复并复用缓存。
- 训练入口验证审核通过样本数等于实际训练样本数。
