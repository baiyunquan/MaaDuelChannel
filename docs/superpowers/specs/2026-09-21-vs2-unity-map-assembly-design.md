# VS-2 Unity 地图组装器设计

状态：用户已确认（2026-09-21）

## 目标

制作一个可重复运行的离线地图组装器，读取 VS-2 关卡数据与 Windows 版游戏资源，在 Unity 中按格子数据实例化地砖、环境 prefab 和红蓝起终点特效，并使用该关卡的相机配置渲染成品。

首个目标关卡为 `activities/act1enemyduel/level_act1enemyduel_02a`。输出为不含角色和 UI 的战场底图，保留地面、场景环境、红蓝路径和起终点。渲染结果供视频坐标标定和后续视觉数据标注使用。

不把现成视频帧或 PRTS 预览图当作背景拼贴；视频帧只用来核对构图、路径位置和镜头视角。程序构建的场景由原始关卡数据和游戏资源驱动。

## 已核实输入

- 游戏 AssetBundle 目录：`E:\Program Files\Arknights\Arknights_Data\StreamingAssets\AB\Windows`。
- AssetBundle 使用 Unity `2021.3.39f1`；渲染工程锁定该 Editor 版本。
- 关卡数据来自本地 `ArknightsResource-50` 仓库的 `level_act1enemyduel_02a.json`。`mapData.map` 是 11 行、每行 15 个索引；`mapData.tiles` 有 165 项，`tileKey` 分布为 99 个 `tile_infection_dqq`、48 个 `tile_forbidden`、9 个 `tile_start_dqq` 和 9 个 `tile_end_dqq`。
- 同仓库 `map_camera_views/maps.json` 有该关卡的相机参数；`map_camera_views/summary.json` 标注主题为 `LMC_DAY`。
- 已找到地砖资源包 `battle/prefabs/[uc]tiles.ab`，以及独立的路径/地块特效资源包 `battle/prefabs/effects/tile.ab`。后者包含 `tile_start.prefab`、`tile_end.prefab` 和多种 `tile_end_multiplayer_*` prefab。
- 已提取的场景资源包括 `arts/maps/map_battleground`、`arts/maps/multi_player`、`arts/maps/map_lm_center` 和 `arts/maps/common`。实际装配时按关卡主题和资源依赖清单筛选，不把所有地图目录无条件叠加。
- 参考视频帧为 1920×864，战场上可见浅色地砖、红色路线和青蓝色路线。该帧有角色与 UI，验收对比时仅使用无遮挡的战场区域。

## 方案

### 执行环境

使用 Unity Editor `2021.3.39f1` 创建最小化的离线渲染工程。Editor 加载游戏 Windows AssetBundle，保留 prefab、mesh、材质和贴图引用，按资源依赖关系装入场景。此次只需要 Windows Editor 本身，不安装 Android、iOS、WebGL 等平台构建模块。

渲染工程和 C# 装配逻辑存入 MaaDuelChannel 子模块。AssetBundle、解包缓存和渲染结果留在外置 workspace，不提交进 Git。工程版本文件记录 Editor 版本；Unity 生成的 `Library`、缓存和临时场景加入忽略规则。

### 装配流程

1. Python CLI 接收游戏资源根目录、关卡 JSON、相机配置、输出目录和输出分辨率，并生成显式配置文件供 Unity 批处理入口读取。
2. Unity 装配器读取关卡 `mapData.map` 与 `mapData.tiles`。每个网格元素先作为 tile 索引解析，再根据 tile 的 `tileKey`、`heightType` 等属性选择对应 prefab 和高度；保留源行列坐标，场景投影方向通过相机参数及视频参照确认。
3. 资源解析器读取主题、AssetBundle 清单和跨包依赖，载入 VS-2 的环境 mesh、共享贴图和材质。首要解析主题为 `LMC_DAY`；候选资源包包括 `map_lm_center`、`map_battleground`、`multi_player`、`common`，最终装载集合以实际 prefab 引用闭包为准。
4. 对 `tile_start_dqq` 和 `tile_end_dqq` 实例化正确的多人对决 prefab 变体。地砖实例与起终点/路径特效分层管理，不将特效贴图作为基础地砖。使用统一的固定时间点渲染动画，确保结果可重复；另输出不含路径特效的地面图层。
5. 根据 `map_camera_views/maps.json` 与 `summary.json` 重建相机。先按原值构造相机，再依据参考视频的可见场地边界核对视野、朝向、投影和裁切；不通过拉伸 PRTS 预览图校正镜头。
6. 在 Unity 离屏相机中渲染颜色图和语义图层，导出到外置 workspace。

### 命令与产物

在现有 `duel` CLI 下新增地图组装入口，建议形式：

```text
duel assemble-map --stage-json <path> --camera-json <path> --game-assets <path> --output-dir <path>
```

默认输出 1920×864，并允许覆盖分辨率、采样动画时间和地图 key。每次运行生成：

- `vs2-02a.png`：包含环境、地面、红蓝路径的无角色/无 UI 地图。
- `ground.png`：不含路线和起终点特效的地面/环境图层。
- `red-route-mask.png` 与 `blue-route-mask.png`：分别对应红方和蓝方路径的二值语义遮罩。
- `assembly-manifest.json`：关卡 key、输入文件哈希、Unity 版本、AssetBundle 及依赖清单、tile prefab 映射、相机参数、输出尺寸和固定动画时间。
- `grid-projection.json`：15×11 格子的投影坐标、行列原点约定和相机标定信息，供视频标定和人工审核工具使用。

产物默认写到用户指定的 workspace 输出目录。程序源码和配置样板入 Git，游戏原始资源、渲染图和 Unity Library 不入 Git。

## 一致性与验收

- 对每个网格位置按 JSON 索引实例化正确 tile；`tileKey` 映射明确记录，不以数组顺序猜测。
- 9 个红方起点和 9 个蓝方终点完整可见，方向、颜色、边界位置和路径形状与参考视频战场一致。
- 输出同时包含环境底图、无路线地面图、红蓝独立遮罩和网格投影元数据。
- 同一组输入重复执行时，除 Unity 生成的无关元数据外，图像输出保持一致；动画以固定时间采样。
- 颜色图的构图和地砖透视以 1920×864 游戏画面战场区域为参照，不将 UI、选手头像或角色烘焙进背景。
- 缺失资源依赖、tileKey 映射或相机字段时，输出明确错误和资源名，不静默使用错误贴图或任意拉伸。
- 采用原始游戏 prefab、材质、mesh 和贴图；如果存在 Unity 内部关卡装配代码专用的运行时参数，以对照画面验证并把实际参数写入 manifest。

## 边界与风险

- 游戏内部关卡加载器源码不可用。本工具复现可从关卡数据、Prefab 和资源依赖中确认的装配流程，不会直接调用游戏私有运行时方法。
- `tile_start_dqq`、`tile_end_dqq` 到多人对决 prefab 变体的具体映射需结合 AssetBundle 对象引用和参考帧核对，不能仅凭 prefab 名称选取。
- 若资源引用了独立着色器/材质包，装配器需沿 Bundle 依赖清单继续加载；只有在确认原 Shader 无法离线加载后，才设计兼容渲染替代方案，并在产物清单标注。
- 相机 profile 中的向量含义和格子行列原点需通过资料结构与像素对照确认，禁止在数据解析层提前硬编码翻转。
- 游戏原始资源只在用户本机读取。MaaDuelChannel 仅存放可复现的程序、样板配置和文档，不复制发行版资产。
