# 战斗知识增强关系 Transformer 实施计划

## 目标

在现有 `DuelTransformer` 训练链路中加入 VS-2 地图数值、攻击属性和技能机制，并让双方单位在编码阶段发生关系交互。保持左右交换后 logit 严格反对称，继续使用全部审核通过样本训练。

## 约束

- 地图数值以 PRTS `VS-2 争锋对决！` 页面为准。
- 敌人页只补充攻击方式、伤害类型、天赋和技能；每条数据保留来源与修订号。
- 自动解析的技能机制携带审核状态和已知性掩码，缺失信息不解释为“没有机制”。
- 预测输入仅包含开战前可见阵容、站位和静态战斗知识。
- 本轮只进行单元测试与一次 CPU 小批量前向/反向验证，不启动正式训练。

## Task 1：战斗知识契约与 PRTS 采集

新增 `combat.py` 与 `prts_combat.py`，定义可版本化的敌人数值、攻击属性、技能机制和地图规则；从 PRTS API 解析 VS-2 的 58 行敌人数据，并关联现有素材目录 ID。

验证：`pytest tests/assets/test_combat_knowledge.py -q`

## Task 2：特征编码与数据批处理

将知识表编码为固定维度数值和机制向量；批处理同时输出原始战场坐标、双方知识特征、知识有效掩码，未知敌人使用显式未知向量。

验证：`pytest tests/training/test_model.py -q`

## Task 3：关系 Transformer

用双方联合编码替换独立队伍编码。注意力关系偏置同时接收距离、攻击范围、伤害对防御、控制对免疫以及友军减防/减抗协同特征；最终使用 `g(left,right)-g(right,left)` 保持反对称。

验证：`pytest tests/training/test_model.py -q`

## Task 4：训练、模型契约与 CLI

训练入口加载知识表并校验哈希，把特征版本、知识版本和知识哈希写入 checkpoint、训练报告和模型版本契约；新增 PRTS 战斗知识同步命令。

验证：`pytest tests/training/test_predictor_training.py tests/test_cli.py -q`

## Task 5：工作区数据与基本回归验证

将完整 VS-2 知识表写入外部 workspace，核对敌人数量、数值来源、缺失页面和 ID 映射报告；运行相关测试和全套短测试。

验证：`pytest -q`
