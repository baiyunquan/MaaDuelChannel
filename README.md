# MAA Duel Channel

从《明日方舟》争锋频道录屏中提取绿藤城对局，训练初始阵容胜负预测模型。

当前仓库独立于 MaaAssistantArknights。等仓库拥有可克隆的远程地址后，再把它挂载为 MAA 的 tools/MaaDuelChannel submodule。

## 当前范围

- 数据源：D:\MAA-DuelChannel\training-data
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
uv sync --extra cu128 --extra ocr --extra vision --group dev
~~~

仅运行 CPU 测试和小规模训练：

~~~powershell
uv sync --extra cpu --group dev
~~~

cpu 与 cu128 不能同时启用。

## 工作目录

训练视频目录保持只读。默认示例：

~~~text
D:\MAA-DuelChannel\
├── training-data\          # 原始视频，只读
└── workspace\
    ├── assets\             # 可追溯的头像、动画和背景
    ├── synthetic\          # YOLO 合成训练集
    ├── manifests\          # 视频、自动识别和 Predictor JSONL
    ├── frames\             # 准备、站位和胜负证据帧
    ├── review\platform\    # Platform 上传包、裁剪图和本地映射
    ├── annotations\        # Platform 回导的不可变标注版本
    ├── datasets\           # Hard/Replay/Base 采样后的训练版本
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
uv run duel assets fetch-prts --catalog D:\MAA-DuelChannel\workspace\manifests\greenvine-prts-catalog.csv --workspace D:\MAA-DuelChannel\workspace
~~~

目录结构为 `assets/portraits/<id>/thumbnail.png` 和 `assets/battlefield_spine/<id>/variant-*/`。后者每个战斗姿态单独保存 `.skel`、`.atlas`、纹理，并在 `assets/catalog.json` 中记录 PRTS 页面/文件 URL 与 SHA-256；重复运行会复用同 URL 的本地文件，`--force` 可强制更新。当前目录含 91 类头像和战场 Spine，其中 ID 79–91 补齐 VS-2 敌人表中原目录遗漏的 13 类。

同步 VS-2 地图数值、敌人攻击属性、天赋和技能：

~~~powershell
uv run duel assets fetch-prts-combat --workspace D:\MAA-DuelChannel\workspace
~~~

结果写入 `assets/combat/vs2_enemy_combat.json`。生命、攻击、防御、法抗、攻击间隔、重量、移速和攻击范围直接取自 VS-2 地图表；敌人页只补充攻击方式、伤害类型和技能。表格保留地图与敌人页的 PRTS 修订号、原始数值文本、技能原文和自动解析状态。

PRTS 页面说明游戏图片、动画等版权归鹰角网络及其关联公司所有。此命令只把素材写入外部 workspace，不会把图片或模型加入 Git；见 [PRTS 版权说明](https://prts.wiki/w/PRTS:%E7%89%88%E6%9D%83)。

### 解包本机 PC 客户端资源

`vendor/Ark-Unpacker` 是独立的 Git submodule，并递归包含其 Ark-FBS-Py 依赖。初始化后运行：

~~~powershell
git submodule update --init --recursive vendor/Ark-Unpacker
uv run python scripts/unpack_game_assets.py `
  --game-root "E:\Program Files\Arknights" `
  --workspace D:\MAA-DuelChannel\workspace
~~~

脚本读取 PC 客户端的 `Arknights_Data/StreamingAssets/AB/Windows` 和 `Arknights_Data/PersistentData/Bundles`，同名热更新包覆盖初始包。它只暂存并解包 `battle` Spine、`spritepack/icon_enemies*`、`ui/enemyduel` 和 `arts/ui/stage_mappreview_*_duel_*`，不扫描或解包整个 24 GB 游戏资源目录。Ark-Unpacker 的 Python 3.12 环境和依赖锁保存在 workspace 的 `tools/ark-unpacker` 下。

每次解包输出到 `assets/game_assets/runs/<UTC时间>/`，包含独立的 `battlefield_spine`、`roster_icons`、`duel_ui`、`duel_stage_previews` 目录及哈希清单。脚本用 PRTS Spine 文件名尝试匹配游戏战斗模型并记录结果；它不会把选手头像和战斗模型混写，也不会覆盖 PRTS 资源目录。Duel 预览图按客户端原始包名保存为候选素材，只有核实与 VS-2 地图对应后再登记为该地图背景。

游戏文件及解包结果只保存在本地 workspace，不加入 Git。脚本、依赖锁和第三方工具 submodule 保存在 MaaDuelChannel 仓库。

准备一张或多张无单位的绿藤城背景，然后同步素材：

~~~powershell
uv run duel assets sync --catalog C:\path\to\green-vine-catalog.csv --background-dir C:\path\to\green-vine-backgrounds --workspace D:\MAA-DuelChannel\workspace
~~~

workspace\assets\catalog.json 会列出缺少头像或动画的敌人。涉及缺失类别的自动样本需要补素材或人工审核。

当前 `synth` 可以读取 `animation` 图片/视频；它还不会把 `.skel`/`.atlas` 渲染成动画帧。PRTS Spine 资源先作为独立、可追溯的源文件保存，不能当成选手缩略图或普通静态图片送入合成器。

## 完整流程

### 1. 扫描视频

~~~powershell
uv run duel scan --input-dir D:\MAA-DuelChannel\training-data --workspace D:\MAA-DuelChannel\workspace
~~~

扫描使用 ffprobe 获取尺寸、帧率和时长，并计算视频 SHA-256。未变化的文件会复用已有记录。当前训练只消费识别为 green_vine 的视频。

### 2. 生成合成 YOLO 数据

~~~powershell
uv run duel synth --workspace D:\MAA-DuelChannel\workspace --portrait-variants 40 --detection-images 1000 --seed 20260920
~~~

头像和战场数据都只生成一套训练样本。Ultralytics classification 初始化时强制要求 `val` 目录，因此头像数据用硬链接（不支持时复制）建立与 train 完全相同的兼容视图；战场数据的 val 也指向 train。它们都不是留出集，报告中的验证结果仍然只能视作训练集拟合结果。每次生成先写入临时目录，完成后整体替换 synthetic，旧类别和旧图片不会混入新数据。

### 3. 训练两个视觉模型

~~~powershell
uv run duel train-vision roster --workspace D:\MAA-DuelChannel\workspace --epochs 100 --device 0 --batch -1 --cache disk --amp --all
uv run duel train-vision battlefield --workspace D:\MAA-DuelChannel\workspace --epochs 100 --device 0 --batch -1 --cache disk --amp --all
~~~

- roster：准备区圆形头像分类。
- battlefield：倒计时归零后、ROUND 遮罩出现前的敌人检测。
- 数量由 RapidOCR 识别。
- 战场框的底边中心作为单位站位。
- Ultralytics 基础权重下载到 workspace\models\base，不会写入命令的当前目录。

### 4. 提取对局

~~~powershell
uv run duel extract --input-dir D:\MAA-DuelChannel\training-data --workspace D:\MAA-DuelChannel\workspace --device 0 --half --batch-size 32
~~~

提取器以低帧率扫描时间线，再只在证据时间点运行昂贵模型：

1. OCR 找到倒计时和 ROUND XXX。
2. 连续准备帧投票得到双方类型与数量。
3. 选择倒计时消失到 ROUND 遮罩出现之间最清晰的站位帧。
4. YOLO 检测结果按双方和 (enemy_id, count) 严格核对。
5. 连续追踪橙色与蓝色血条，稳定消失的一方判负。
6. 高置信且完全一致的样本自动 accepted；其余样本进入 pending。

自动结果写入 manifests\rounds.auto.jsonl。不完整窗口和单局识别错误写入 reports\extraction-errors.json，同视频中已经成功提取的局仍会保留。重复提取可以覆盖自动结果，不会覆盖 review\corrections.jsonl 中的人工修正。

### 5. 在 Ultralytics Platform 人工审核

~~~powershell
uv run duel review --workspace D:\MAA-DuelChannel\workspace
~~~

命令会打开 Ultralytics Platform，并在 `review\platform\<export-id>` 生成三个上传包：

- `roster_classification.zip`：六个准备区头像裁剪，使用 Platform 的图像分类选择器纠正敌人类型。
- `battlefield_detection.zip`：完整战场帧和初始 YOLO 框，使用矩形工具增删、移动框并纠正类别。
- `ocr_classification.zip`：六个数量数字裁剪，以 `count_1`、`count_2` 等类别纠正 OCR；`empty` 和 `unreadable` 单独保留。

三个数据集在 Platform 完成审核后，从数据集或 Versions 页分别下载 NDJSON，并一次导回：

~~~powershell
uv run duel annotate import `
  --workspace D:\MAA-DuelChannel\workspace `
  --export-directory D:\MAA-DuelChannel\workspace\review\platform\platform-20260921T120000Z `
  --roster C:\Downloads\roster.ndjson `
  --battlefield C:\Downloads\battlefield.ndjson `
  --ocr C:\Downloads\ocr.ndjson `
  --version annotations-v1
~~~

文件名中的稳定标注 ID 会把 Platform 结果映射回视频、局、左右方和槽位。导入器同时写入
`annotations\versions\annotations-v1`，并将头像、数量和框合并到 `review\corrections.jsonl`。已有胜负标签且双方数量与检测框完全一致时，该局自动成为 accepted；否则继续保持 pending。

样本身份由视频 SHA-256 和局序号决定，不依赖证据帧的具体时间。某局在一次自动重提取中暂时缺失时，已经保存的人工审核结果仍会进入有效数据集。

### 6. 构建 Hard、Replay、Base 视觉训练集

~~~powershell
uv run duel annotate sample `
  --workspace D:\MAA-DuelChannel\workspace `
  --annotation-version annotations-v1 `
  --output-version vision-v2 `
  --hard-fraction 0.5 `
  --replay-fraction 0.3 `
  --base-fraction 0.2
~~~

- Hard Set：全部经人工修改的真实样本，始终保留，不受最大样本数截断。
- Replay Set：按任务和类别轮转抽取的“模型原识别正确”真实样本，用于抑制灾难性遗忘。
- Base Set：从原始合成头像与战场数据中分层抽样，维持类别覆盖和基础视觉能力。

采样结果写入 `datasets\vision-v2`，同一个标注不会同时进入多个集合。随后针对审核数据进行下一轮训练：

~~~powershell
uv run duel train-vision roster --workspace D:\MAA-DuelChannel\workspace --dataset-version vision-v2 --device 0 --batch -1 --amp --all
uv run duel train-vision battlefield --workspace D:\MAA-DuelChannel\workspace --dataset-version vision-v2 --device 0 --batch -1 --amp --all
uv run duel train-vision ocr --workspace D:\MAA-DuelChannel\workspace --dataset-version vision-v2 --device 0 --batch -1 --amp --all
~~~

训练出 OCR 分类模型后，`extract` 会优先批量使用它识别准备区数量；倒计时和 `ROUND` 文本仍由 RapidOCR 识别。没有 OCR 分类 checkpoint 时自动回退到 RapidOCR 数量识别。

### 7. 构建 Transformer 数据集

~~~powershell
uv run duel build-dataset --workspace D:\MAA-DuelChannel\workspace
~~~

构建前必须准备 `assets\calibration\green-vine-video-v1.json`。视频标定使用至少 8 个地砖交点拟合单应矩阵，并用至少 2 个独立点检查；最大检查误差超过 0.15 格时停止。自动检测的归一化底边中心会同时保存为原始坐标，并转换成以格为单位的地面坐标。accepted 样本缺少标定时会列出样本 ID 并停止。

输出 manifests\predictor.jsonl 和 predictor.meta.json。构建器先用人工 correction 覆盖自动结果，只写 accepted 样本，并记录数据集 SHA-256、实际写入数量、胜负分布，以及知识、公式、特征结构、标定和 ID 词表的版本与 SHA-256。数据集版本由这些依赖共同生成，知识变化不会覆盖旧版本。

### 8. 训练胜负预测模型

~~~powershell
uv run duel train-predictor --workspace D:\MAA-DuelChannel\workspace --epochs 100 --batch-size 64 --device cuda --workers 4 --amp --amp-dtype float16 --pin-memory --tf32 --seed 20260920 --all
~~~

每只单位是一个 token，包含敌人 ID、地面格坐标、VS-2 基础值、开场攻击与技能数值、后续形态摘要，以及 12 个聚集、离散、前后排、承伤和保护阵型量。数量由同类 token 的重复次数表示。未知知识使用显式掩码，不会被解释成数值为零或没有技能。

双方单位进入同一个 Relation Transformer。有向关系矩阵固定为 20 维：距离、前后与横向位移、射程余量、接敌时间、三类有效 DPS、单次普攻、普攻次数、持续 TTK、10 秒开场伤害、群攻覆盖、硬控覆盖、目标分配、治疗、减防减抗协同、前排遮护和保护秒数。物理伤害按每段分别扣防，目标与治疗容量守恒。最终 logit 为 `g(left, right) - g(right, left)`，因此交换双方时预测概率严格互补。

产物：

- models\predictor\last.pt
- models\predictor\best-train-loss.pt
- models\predictor\training-report.json
- models\predictor\versions\<model-version>\best-train-loss.pt

训练入口会验证 accepted_samples == written_samples == training_samples，并核对数据集记录的知识、公式、特征结构、标定和 ID 词表 SHA-256；不一致时停止。checkpoint、训练报告和模型版本契约保存同一组依赖。

`--amp-dtype bfloat16` 可在支持 BF16 的显卡上使用；`--compile` 可选择启用 `torch.compile`，首次编译会增加启动时间。DataLoader 使用 pinned memory 和 non-blocking GPU 传输。YOLO 训练支持自动批量大小或显存占比；抽取阶段会批量识别同局的头像裁剪，并对检测器暴露批量推理接口。

### 9. 推理与关系解释

推理输入使用不带胜负标签的 `BattleState` JSON。`x`、`y` 是地面格坐标，必须同时提供与 checkpoint 一致的 `calibration_id` 和 `calibration_sha256`：

~~~powershell
uv run duel predict-duel --workspace D:\MAA-DuelChannel\workspace --input D:\MAA-DuelChannel\battle-state.json --device cuda --explain --output D:\MAA-DuelChannel\prediction.json
~~~

输出包含左右获胜概率。`--explain` 额外导出带单位实例 ID 的 DPS、TTK、10 秒爆发、群攻覆盖、目标分配、遮护分数和保护秒数矩阵。

### 10. 生成报告

~~~powershell
uv run duel report --workspace D:\MAA-DuelChannel\workspace
~~~

报告位于 reports\summary.json、summary.md 和 extraction-errors.json。

## 数据契约

RoundSample 保存视频哈希、局序号、四个阶段时间戳、三张证据帧、双方 roster、双方 unit、胜方、置信度、审核状态和流水线版本。

数据与模型使用 `contract_version = 1.0.0`：

- Platform 导出清单：`review\platform\<export-id>\annotation-manifest.jsonl`
- 人工标注版本：`annotations\versions\<version>\dataset.json`
- 训练数据版本：`datasets\<version>\dataset.json`
- YOLO 模型版本：`models\vision\<task>\<model-version>\model.json`
- Transformer 模型版本：`models\predictor\versions\<model-version>\model.json`

契约记录父版本、数据清单 SHA-256、任务与采样桶计数、基础权重、训练参数、精度、设备、checkpoint SHA-256 和 Git commit。`models\vision\<task>\weights\best.pt` 始终是当前部署副本，历史 checkpoint 保留在模型版本目录中。

`RoundSample` 保留相对有效游戏视口归一化到 [0, 1] 的原始检测坐标：

~~~text
bbox = (x1, y1, x2, y2)
position = ((x1 + x2) / 2, y2)
~~~

也就是使用检测框底边中心作为初始落点。构建 `PredictorSample` 时再由绑定的视频标定转换到地面格坐标；原始坐标、位置质量、标定 ID 和 SHA-256 一并保留。PRTS 预览图不能复用视频标定。

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
