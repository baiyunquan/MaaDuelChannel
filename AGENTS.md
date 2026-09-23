# MaaDuelChannel 开发指南

## 项目定位

本项目从《明日方舟》“争锋频道”录屏中提取绿藤城（`green_vine`）对局，经过视觉识别、人工审核和数据版本化，训练根据初始阵容与站位预测左方胜率的 Relation Transformer。

- 输入：准备阶段敌人类型/数量、倒计时结束时的初始站位。
- 标签：战斗结束时的稳定存活方；血条或存活检测只生成标签，不进入 Predictor。
- 当前训练不划分验证集/测试集，指标仅代表训练集拟合，不能称为泛化精度。
- 当前不包含 MAA 实时控制、在线推理接入或正式发布权重。

## 仓库与数据边界

- Git 仓库是代码与小型配置的来源；大型资源位于 `/mnt/data/MAA-DuelChannel/`。
- `/mnt/data/MAA-DuelChannel/training-data/` 是原始视频，只读。不要重命名、移动、覆盖或写入。
- `/mnt/data/MAA-DuelChannel/workspace/` 保存素材、帧、标注、数据集、模型和报告，可由流水线更新。
- 不要把视频、模型权重、解包游戏资源或 workspace 产物加入 Git。
- `PipelineConfig` 会拒绝把 workspace 放进原视频目录；不要绕过该保护。
- 工作树可能包含用户正在进行的修改。编辑前先看 `git status --short`，不要顺手格式化、回退或覆盖无关文件。

截至 2026-09-23，外部 workspace 已登记 358 个视频（69 个 `green_vine`）、91 类素材和 449 个待审核局；已有 roster/battlefield 视觉权重，尚无正式 Predictor 权重。这只是本机快照，运行前以清单和模型目录为准。

## 代码结构

| 路径 | 职责 |
|---|---|
| `src/maa_duel/cli.py` | Typer 命令入口；命令应保持薄层 |
| `config.py`, `schema.py`, `contracts.py`, `store.py` | 路径配置、Pydantic 数据契约、版本契约、原子 JSONL I/O |
| `video/` | ffprobe 扫描、有效视口、准备/战斗阶段状态机 |
| `vision/` | OCR、卡槽与阵容识别、战场检测核对、胜负判定 |
| `extraction.py` | 串联视频阶段、视觉模型、证据帧和 `RoundSample` |
| `extraction_state.py` | 不可变提取 run、active 指针和原子发布 |
| `assets.py`, `prts_assets.py`, `prts_combat.py` | 可追溯素材、PRTS Spine 与 VS-2 战斗知识 |
| `synthetic.py` | 生成 roster 分类和 battlefield 检测训练集 |
| `annotations.py`, `sampling.py`, `review.py`, `gui/` | 标注导入导出、Hard/Replay/Base 采样、本地审核界面 |
| `calibration.py`, `dataset.py` | 单应标定及 `RoundSample` 到 Predictor 数据转换 |
| `combat.py`, `training/derived.py`, `training/features.py` | 战斗知识、派生关系矩阵和模型输入特征 |
| `training/model.py`, `predictor.py`, `inference.py`, `vision.py` | Transformer/YOLO 训练、推理与版本产物 |
| `tests/` | 与源码职责对应的 pytest 测试 |
| `scripts/` | 游戏资源解包等独立工具 |
| `vendor/` | 固定版本子模块；除升级依赖外不要修改 |

主数据流：`scan -> assets/synth/train-vision -> extract -> review/annotate -> build-dataset -> train-predictor -> predict-duel/report`。具体参数与操作步骤以 `README.md` 为准。

## 环境配置

要求 Python `>=3.12,<3.14`、`uv`、`ffmpeg`/`ffprobe`。正式训练建议 NVIDIA GPU 与 CUDA 12.8；CPU 与 `cu128` extra 互斥。

```bash
# 基础开发/CPU 测试
uv sync --extra cpu --group dev

# 完整 NVIDIA 视觉、OCR 和审核环境
uv sync --extra cu128 --extra ocr --extra vision --extra review --group dev

git submodule update --init --recursive
uv run duel --help
```

当前 Linux 工作站可直接使用 `/mnt/data/MAA-DuelChannel/`；README 中的 `D:\MAA-DuelChannel` 是 Windows 等价示例。命令应显式传入 `--input-dir` 和 `--workspace`，不要依赖当前目录猜测数据位置。

## 关键约束

- `RoundSample` 的身份由视频 SHA-256 与局序号稳定生成，不应依赖证据帧时间。
- 检测坐标先保存为有效游戏视口内 `[0, 1]` 归一化框；单位位置取框底边中心。
- Predictor 数据必须绑定视频标定的 ID 与 SHA-256；PRTS 地图图不能复用视频标定。
- accepted 样本必须有胜者、双方非空阵容，且 roster 数量与 unit 数量严格一致。
- 所有消费者通过 active 提取指针读取同一 run；`--force --reset-review` 只切换到空审核的新 run，旧 corrections 保留归档。
- 数据/模型契约当前为 `contract_version = 1.0.0`。修改 schema、特征、公式、知识或标定时，同步版本与哈希校验。
- 写清失败原因并保留同视频中成功的局；不要因局部识别失败丢弃整段视频。
- 文件输出优先采用临时文件加原子替换；路径与 JSON/JSONL 使用 UTF-8。

## 开发与验证

- Python 采用 3.12 语法、类型标注、Pydantic v2；Ruff 行宽 120，`vendor/` 被排除。
- 新功能或修复应在相应 `tests/` 子目录添加最小回归测试；视觉逻辑使用小型合成帧/夹具，避免测试依赖完整视频或网络。
- 先运行目标测试，再运行完整检查：

```bash
uv run pytest tests/path/to/test_file.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

- 不要用全量训练证明普通代码修改；训练入口可用短 epoch/小夹具做冒烟验证。
- 修改 CLI、数据契约或流水线顺序时同步更新 `README.md`；不要在本文件复制完整命令手册。
