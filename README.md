# MAA Duel Channel

从《明日方舟》争锋频道录屏中提取绿藤城对局，训练初始阵容胜负预测模型。

当前仓库独立于 MaaAssistantArknights。等仓库拥有可克隆的远程地址后，再把它挂载为 MAA 的 tools/MaaDuelChannel submodule。

## 当前范围

- 数据源：G:\MAA-DuelChannel\training-data
- 当前场地：绿藤城（Ivyvine）
- 已确认素材：69 段绿藤城视频，约 4.69 小时
- 输入：准备阶段的敌人类型、数量和倒计时归零时的初始站位
- 标签：战斗结束前橙色、蓝色血条的稳定存活状态
- 输出：左方获胜概率
- 训练策略：所有审核通过样本都参与训练，不划分验证集或测试集

血条只用于从视频生成胜负标签，不作为 Transformer 输入。报告中的准确率和损失都是训练集拟合指标，不能解释为泛化精度。

## 环境

要求：Windows、Python 3.12、uv、ffmpeg/ffprobe。NVIDIA GPU 正式训练建议使用 CUDA 12.8 环境。

~~~powershell
uv sync --group dev
uv sync --extra cu128 --extra ocr --extra vision --extra review --group dev
~~~

仅运行 CPU 测试和小规模训练：

~~~powershell
uv sync --extra cpu --group dev
~~~

cpu 与 cu128 不能同时启用。

## 工作目录

训练视频目录保持只读。默认示例：

~~~text
G:\MAA-DuelChannel\
├── training-data\          # 原始视频，只读
└── workspace\
    ├── assets\             # 可追溯的头像、动画和背景
    ├── synthetic\          # YOLO 合成训练集
    ├── manifests\          # 视频、自动识别和 Predictor JSONL
    ├── frames\             # 准备、站位和胜负证据帧
    ├── review\             # 人工 correction
    ├── models\             # YOLO 与 Transformer checkpoint
    └── reports\            # 统计和错误报告
~~~

代码会拒绝把 workspace 放到原始视频目录内部。

## 素材目录

CSV 或 JSON 必须为每个敌人提供稳定的正整数 id。CSV 支持以下字段：

| 含义 | 支持的字段名 |
|---|---|
| 显示名 | name、名称 |
| 原始名 | original_name、原始名称 |
| 选手缩略图 | portrait、portrait_url、头像 |
| 可直接合成的动画/精灵图 | animation、animation_url、动画 |
| PRTS Spine 查询名 | original_name、原始名称 |

选手缩略图与战场模型是不同素材：`portrait` 只用于准备阶段头像分类；`battlefield_spine` 保存战场 Spine 骨骼、atlas 和 atlas 引用的纹理；`animation` 仍表示可被合成器直接读取的图片或视频。三者在 manifest 与磁盘目录里分开记录。

`portrait` 和 `animation` 可以是相对 CSV 的本地路径，也可以是 HTTP(S) URL。同步结果记录来源 URI 和 SHA-256。普通素材格式见 examples/catalog.example.csv；PRTS 查询表格式见 examples/prts-catalog.example.csv。仓库不包含 CannotMax 的代码或下载来的游戏素材。

`vendor/ark_info_search` 是固定版本的 Git submodule，提供 PRTS MediaWiki API 的参考实现。该项目采用 AGPL-3.0；MaaDuelChannel 不导入它的 Python 代码，资源下载器在 `src/maa_duel/prts_assets.py` 中单独实现 PRTS 查询和 Spine 文件解析。

把 PRTS 原名（保留名字中的引号和标点）写入 `original_name` 后，运行：

~~~powershell
uv run duel assets fetch-prts --catalog G:\MAA-DuelChannel\workspace\manifests\greenvine-prts-catalog.csv --workspace G:\MAA-DuelChannel\workspace
~~~

目录结构为 `assets/portraits/<id>/thumbnail.png` 和 `assets/battlefield_spine/<id>/variant-*/`。后者每个战斗姿态单独保存 `.skel`、`.atlas`、纹理，并在 `assets/catalog.json` 中记录 PRTS 页面/文件 URL 与 SHA-256；重复运行会复用同 URL 的本地文件，`--force` 可强制更新。PRTS 为 78 个绿藤城敌人提供头像页和 Spine 页面；下载结果会逐项报告缺失项。

PRTS 页面说明游戏图片、动画等版权归鹰角网络及其关联公司所有。此命令只把素材写入外部 workspace，不会把图片或模型加入 Git；见 [PRTS 版权说明](https://prts.wiki/w/PRTS:%E7%89%88%E6%9D%83)。

准备一张或多张无单位的绿藤城背景，然后同步素材：

~~~powershell
uv run duel assets sync --catalog C:\path\to\green-vine-catalog.csv --background-dir C:\path\to\green-vine-backgrounds --workspace G:\MAA-DuelChannel\workspace
~~~

workspace\assets\catalog.json 会列出缺少头像或动画的敌人。涉及缺失类别的自动样本需要补素材或人工审核。

当前 `synth` 可以读取 `animation` 图片/视频；它还不会把 `.skel`/`.atlas` 渲染成动画帧。PRTS Spine 资源先作为独立、可追溯的源文件保存，不能当成选手缩略图或普通静态图片送入合成器。

## 完整流程

### 1. 扫描视频

~~~powershell
uv run duel scan --input-dir G:\MAA-DuelChannel\training-data --workspace G:\MAA-DuelChannel\workspace
~~~

扫描使用 ffprobe 获取尺寸、帧率和时长，并计算视频 SHA-256。未变化的文件会复用已有记录。当前训练只消费识别为 green_vine 的视频。

### 2. 生成合成 YOLO 数据

~~~powershell
uv run duel synth --workspace G:\MAA-DuelChannel\workspace --portrait-variants 40 --detection-images 1000 --seed 20260920
~~~

头像和战场数据都只生成一套训练样本。Ultralytics classification 初始化时强制要求 `val` 目录，因此头像数据用硬链接（不支持时复制）建立与 train 完全相同的兼容视图；战场数据的 val 也指向 train。它们都不是留出集，报告中的验证结果仍然只能视作训练集拟合结果。每次生成先写入临时目录，完成后整体替换 synthetic，旧类别和旧图片不会混入新数据。

### 3. 训练两个视觉模型

~~~powershell
uv run duel train-vision roster --workspace G:\MAA-DuelChannel\workspace --epochs 100 --device 0 --all
uv run duel train-vision battlefield --workspace G:\MAA-DuelChannel\workspace --epochs 100 --device 0 --all
~~~

- roster：准备区圆形头像分类。
- battlefield：倒计时归零后、ROUND 遮罩出现前的敌人检测。
- 数量由 RapidOCR 识别。
- 战场框的底边中心作为单位站位。
- Ultralytics 基础权重下载到 workspace\models\base，不会写入命令的当前目录。

### 4. 提取对局

~~~powershell
uv run duel extract --input-dir G:\MAA-DuelChannel\training-data --workspace G:\MAA-DuelChannel\workspace
~~~

提取器以低帧率扫描时间线，再只在证据时间点运行昂贵模型：

1. OCR 找到倒计时和 ROUND XXX。
2. 连续准备帧投票得到双方类型与数量。
3. 选择倒计时消失到 ROUND 遮罩出现之间最清晰的站位帧。
4. YOLO 检测结果按双方和 (enemy_id, count) 严格核对。
5. 连续追踪橙色与蓝色血条，稳定消失的一方判负。
6. 高置信且完全一致的样本自动 accepted；其余样本进入 pending。

自动结果写入 manifests\rounds.auto.jsonl。不完整窗口和单局识别错误写入 reports\extraction-errors.json，同视频中已经成功提取的局仍会保留。重复提取可以覆盖自动结果，不会覆盖 review\corrections.jsonl 中的人工修正。

### 5. 人工审核

~~~powershell
uv run duel review --workspace G:\MAA-DuelChannel\workspace
~~~

浏览器界面同时显示准备帧、标注后的站位帧和结束证据帧。可以编辑双方清单、单位落脚点和框、胜方以及审核备注。人工接受时会再次验证每一方各类型的清单数量与单位数量完全相等。

样本身份由视频 SHA-256 和局序号决定，不依赖证据帧的具体时间。某局在一次自动重提取中暂时缺失时，已经保存的人工审核结果仍会进入有效数据集。

### 6. 构建 Transformer 数据集

~~~powershell
uv run duel build-dataset --workspace G:\MAA-DuelChannel\workspace
~~~

输出 manifests\predictor.jsonl 和 predictor.meta.json。构建器先用人工 correction 覆盖自动结果，只写 accepted 样本，并记录数据集 SHA-256、实际写入数量和胜负分布。

### 7. 训练胜负预测模型

~~~powershell
uv run duel train-predictor --workspace G:\MAA-DuelChannel\workspace --epochs 100 --batch-size 64 --device cuda --seed 20260920 --all
~~~

每只单位是一个 token，包含敌人 ID embedding 和归一化位置。数量由同类 token 的重复次数表示。批次使用动态 padding 和 attention mask；右方 x 坐标会镜像为从其出生侧观察的坐标。

双方共享 TeamEncoder。最终 logit 为 g(left, right) - g(right, left)，因此交换双方时预测概率严格互补。

产物：

- models\predictor\last.pt
- models\predictor\best-train-loss.pt
- models\predictor\training-report.json

训练入口会验证 accepted_samples == written_samples == training_samples，不一致时停止。

### 8. 生成报告

~~~powershell
uv run duel report --workspace G:\MAA-DuelChannel\workspace
~~~

报告位于 reports\summary.json、summary.md 和 extraction-errors.json。

## 数据契约

RoundSample 保存视频哈希、局序号、四个阶段时间戳、三张证据帧、双方 roster、双方 unit、胜方、置信度、审核状态和流水线版本。

所有坐标相对有效游戏视口归一化到 [0, 1]：

~~~text
bbox = (x1, y1, x2, y2)
position = ((x1 + x2) / 2, y2)
~~~

也就是使用检测框底边中心作为地面站位。

## 测试与格式检查

~~~powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
~~~

测试覆盖真实 ffprobe 视频探测、两种分辨率坐标、多局状态机、OCR、清单核对、血条胜负、人工 correction、数据计数、Transformer 对称性，以及合成素材到一轮训练的端到端冒烟。

## 已知边界

- 本阶段不执行 69 段视频的正式抽取和全量训练，因此不包含正式权重。
- 默认准备区 ROI 根据现有 1920×864 和 1920×1080 绿藤城视频校准；正式运行后应根据 pending 样本报告调整。
- 没有留出数据，无法在本阶段提供泛化准确率。后续素材充足时应按源视频分组建立独立评估集。
- 当前只生成训练工具和离线模型，不包含 MAA 内的实时游戏控制或推理接入。
