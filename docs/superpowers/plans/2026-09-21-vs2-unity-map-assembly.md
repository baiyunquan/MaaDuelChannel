# VS-2 Unity Map Assembler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task. Track steps with checkbox syntax.

**Goal:** Build and run a Unity-backed assembler that reproduces the clean VS-2 battlefield from stage grid data, game prefabs, asset dependencies, and the stage camera profile.

**Architecture:** The existing Typer CLI validates the local inputs and writes a canonical assembly request. A version-pinned Unity Editor project loads Windows AssetBundles, places each grid tile and themed environment asset, renders the color image and semantic masks, then writes an asset/camera manifest. Game files and image outputs stay in the external workspace.

**Tech Stack:** Python 3.12, Typer, Unity Editor 2021.3.39f1, C# Editor scripting, Unity AssetBundle API, PNG and JSON output.

**Spec:** `docs/superpowers/specs/2026-09-21-vs2-unity-map-assembly-design.md`

## Global Constraints

- Use Unity Editor `2021.3.39f1` for the game’s Windows AssetBundles.
- Pass `-noUpm` to Editor batch runs: the installed Package Manager returned HTTP 500 for both this project and a fresh empty project, while this assembler uses no external UPM packages.
- Build `activities/act1enemyduel/level_act1enemyduel_02a` from its 15-column by 11-row map grid.
- Preserve all 165 tile indices and the confirmed counts: 99 `tile_infection_dqq`, 48 `tile_forbidden`, 9 `tile_start_dqq`, 9 `tile_end_dqq`.
- Keep the confirmed endpoint coordinates: `tile_start_dqq` at source column 1 and `tile_end_dqq` at source column 13 for rows 1–9; align red/blue semantics to the reference frame.
- Default render size is 1920×864 and the scene excludes units and UI while keeping the floor, environment, and both route sides.
- Resolve and record required AssetBundle dependencies; use the game’s hotfix overlay when a bundle path is overridden.
- Store source assets, Unity caches, and rendered outputs outside Git. Commit only the assembler, small configuration files, and documentation.
- Do not add or run automated tests in this session. Validate by generating the requested map and inspecting the image, masks, and manifest.

## Review Focus

- **Tile array indexing and row order:** reject out-of-range indices and keep source row/column coordinates; manually check all 165 cells and projected grid corners in Task 2 and Task 7.
- **Hotfix versus StreamingAssets selection:** select the installed hotfix bundle only when the hot-update index declares that path; record resolved path and SHA-256 in Task 3 and Task 7.
- **AssetBundle dependency order:** load shared resources before consuming prefabs and report the exact missing bundle/object; review the load manifest and Unity log in Tasks 3–5.
- **`tileKey` to prefab mapping:** map the four VS-2 keys to exact named prefabs and fail on an unknown key; inspect the constructed 15×11 grid in Task 4.
- **Camera and route side orientation:** do not assume row flips or left/right colors; compare the clean battle viewport to the video frame and inspect each semantic mask in Task 6.

---

### Task 1: Scaffold the version-pinned Unity renderer project

**Files:**

- Create: `unity/map-assembler/ProjectSettings/ProjectVersion.txt`
- Create: `unity/map-assembler/Packages/manifest.json`
- Create: `unity/map-assembler/Assets/Editor/MaaDuelChannel.MapAssembler.Editor.asmdef`
- Modify: `.gitignore`

**Interfaces:**

- Produces: an Editor-only Unity project at `unity/map-assembler`, opened with Unity Editor `2021.3.39f1`.
- Consumes: the installed Editor at `D:\Unity\Hub\Editor\2021.3.39f1\Editor\Unity.exe`.

- [ ] **Step 1: Pin the project Editor version**

Create `ProjectSettings/ProjectVersion.txt` with:

```text
m_EditorVersion: 2021.3.39f1
```

- [ ] **Step 2: Create the package manifest and Editor-only assembly definition**

Use an empty package dependency object so Unity supplies only its built-in modules:

```json
{
  "dependencies": {}
}
```

Set the assembly definition name to `MaaDuelChannel.MapAssembler.Editor` and `includePlatforms` to `Editor`.

- [ ] **Step 3: Ignore Unity-generated files**

Add these repository-relative ignores: `unity/map-assembler/Library/`, `Temp/`, `Obj/`, `Logs/`, `UserSettings/`, `Build/`.

- [ ] **Step 4: Open the empty project in batch mode and inspect the Editor log**

Run:

```powershell
& 'D:\Unity\Hub\Editor\2021.3.39f1\Editor\Unity.exe' -batchmode -quit -noUpm -projectPath 'C:\Users\liaic\Source\Repos\MaaAssistantArknights\tools\MaaDuelChannel\unity\map-assembler' -logFile 'D:\MAA-DuelChannel\workspace\tools\unity-map-assembler\logs\project-import.log'
```

Expected: Editor exits successfully and the project log contains no compile errors.

### Task 2: Parse the stage, camera, and canonical assembly request

**Files:**

- Create: `src/maa_duel/map_assembly.py`
- Modify: `src/maa_duel/cli.py`
- Create: `unity/map-assembler/Assets/Editor/AssemblyInput.cs`

**Interfaces:**

- Produces: `MapAssemblyRequest`, `StageCell`, `StageGrid`, `prepare_assembly_input(request) -> Path`.
- The canonical JSON fields consumed by C# are `stageKey`, `theme`, `width`, `height`, `frameTime`, `grid`, `camera`, `bundleRoot`, `bundleFiles`, and `outputDirectory`.
- Each grid item contains `row`, `column`, `tileIndex`, `tileKey`, `heightType`, `buildableType`, `passableMask`, and `playerSideMask`.

- [ ] **Step 1: Define the Python request and cell records**

`MapAssemblyRequest` contains `stage_json: Path`, `camera_json: Path`, `game_assets: Path`, `output_dir: Path`, `stage_key: str`, `width: int = 1920`, `height: int = 864`, and `frame_time: float = 0.0`.

- [ ] **Step 2: Decode every map index through `mapData.tiles`**

Load `mapData.map` as rows of integer indices. For each `(row, column, tile_index)`, index `mapData.tiles[tile_index]`; preserve source coordinates. Reject non-rectangular maps, invalid indices, or tile records without `tileKey`.

- [ ] **Step 3: Read the camera profile and sibling theme summary**

Select the exact `stage_key` from `maps.json`. Read `summary.json` beside it to resolve `theme`; fail if the key is absent from either input. Do not flip rows or reinterpret the profile vectors in Python.

- [ ] **Step 4: Write canonical Unity input JSON**

Write UTF-8 JSON under a temporary run directory inside `output_dir`, including input file hashes and the resolved bundle paths. Return the config path from `prepare_assembly_input`.

- [ ] **Step 5: Add the `duel assemble-map` command**

Add options `--stage-json`, `--camera-json`, `--game-assets`, `--output-dir`, `--stage-key`, `--width`, `--height`, `--frame-time`, `--dry-run`, and optional `--unity-editor`. Have the command call `prepare_assembly_input` and then the Unity process runner; keep imports local as existing commands do. With `--dry-run`, print the parsed grid and resolved bundle plan without launching Unity.

### Task 3: Resolve the asset set and hotfix overlay

**Files:**

- Create: `src/maa_duel/map_assets.py`
- Create: `config/map-assembly-assets.json`
- Modify: `src/maa_duel/map_assembly.py`

**Interfaces:**

- Produces: `BundleFile`, `MapAssetPlan`, `resolve_map_assets(game_assets: Path, theme: str, stage: StageGrid) -> MapAssetPlan`.
- `MapAssetPlan` contains ordered `bundle_files`, a `tile_prefab_by_key` mapping, environment roots, and per-file relative path plus SHA-256.
- Consumes: the installed game’s `StreamingAssets/AB/Windows` and `PersistentData/Bundles` trees.

- [ ] **Step 1: Declare the required tile and effect assets**

Record these exact bundle paths in `config/map-assembly-assets.json`: `battle/prefabs/[uc]tiles.ab` and `battle/prefabs/effects/tile.ab`. Map the stage keys to prefab paths under `dyn/battle/prefabs/[uc]tiles/`: `tile_infection_dqq`, `tile_forbidden`, `tile_start_dqq`, and `tile_end_dqq`.

- [ ] **Step 2: Declare the VS-2 environment resource roots**

Add theme `LMC_DAY` with the candidate roots `arts/maps/map_lm_center`, `arts/maps/map_battleground`, `arts/maps/multi_player`, and `arts/maps/common`. Enumerate each needed `res.ab`, mesh, and trap bundle from the extracted game inventory; do not load unrelated map themes.

- [ ] **Step 3: Resolve hotfix precedence and verify each selected bundle**

Read `PersistentData/Bundles/hot_update_list.json`; when it names a selected relative bundle path, use the corresponding hotfix file, otherwise use StreamingAssets. Reject missing files and compute SHA-256 for every resolved path.

- [ ] **Step 4: Resolve dependencies in load order**

Read the `.idx` resource manifest from each asset root (`StreamingAssets/AB/Windows/*.idx` and `PersistentData/Bundles/*.idx`). Parse its `ResourceManifest.Bundles` and `ResourceManifest.AssetToBundleList` tables with the existing schema in `vendor/Ark-Unpacker/src/fbs/CN/resource_manifest.py`. Follow each bundle’s `AllDependencies` indices to add transitive shared material, texture, and shader bundles ahead of their consumers. Store the resolved order in `MapAssetPlan`; if an index entry or referenced bundle cannot be resolved, report its path and stop rather than skipping it.

- [ ] **Step 5: Inspect the resulting asset plan from the CLI**

Run the `duel assemble-map` command with the local inputs after `--dry-run` is added to the command. Expected summary: theme `LMC_DAY`, 15 columns, 11 rows, 165 cells, tile counts `99/48/9/9`, and existing resolved bundle files with their hashes. Do not launch Unity for dry-run.

### Task 4: Load bundles and instantiate the grid and duel markers

**Files:**

- Create: `unity/map-assembler/Assets/Editor/BundleSet.cs`
- Create: `unity/map-assembler/Assets/Editor/StageGridBuilder.cs`
- Create: `unity/map-assembler/Assets/Editor/DuelMapAssembler.cs`
- Modify: `unity/map-assembler/Assets/Editor/AssemblyInput.cs`

**Interfaces:**

- Produces: `BundleSet.Load(MapAssemblyInput input)`, `StageGridBuilder.Build(MapAssemblyInput input, BundleSet bundles) -> GridBuildResult`, and static Editor entry `MaaDuelChannel.MapAssembler.DuelMapAssembler.Run()`.
- `GridBuildResult` contains `GameObject root`, `Bounds worldBounds`, and a list of cell records pairing row/column/tile index with world position.
- Consumes: canonical input JSON, ordered bundles, per-tile prefab paths, and cell height metadata.

- [ ] **Step 1: Load bundles in the resolved order**

Call `AssetBundle.LoadFromFile` for every path. Retain all loaded bundles until rendering finishes. On a failed load, include the exact path in the thrown Editor error and release previously loaded bundles in `finally`.

- [ ] **Step 2: Resolve exact prefab assets by container path**

Resolve prefab assets with `AssetBundle.LoadAsset<GameObject>(prefabPath)`. Reject absent mappings or non-GameObject assets. Do not fall back from `tile_*_dqq` to a generic tile prefab.

- [ ] **Step 3: Place cells using prefab and game grid transforms**

Create one instance per grid cell, using the source row and column, tile prefab transform, and `heightType`. Determine cell pitch/origin from the original tile prefab and the stage camera profile; keep the source row/column on each object for projection output. Return the root, combined world bounds, and cell/world-position records in `GridBuildResult`.

Check the known endpoint layout before rendering: start tiles are at `(row 1..9, column 1)` and end tiles at `(row 1..9, column 13)` in the source grid.

- [ ] **Step 4: Add multiplayer route and endpoint effects**

Instantiate the multiplayer-specific route/endpoint effects from `battle/prefabs/effects/tile.ab` using the source cell/edge coordinates and the prefab transforms. Choose the `tile_end_multiplayer_*` variant and its orientation only after matching its hierarchy and appearance to the VS-2 reference frame; write the chosen asset path into the output manifest.

- [ ] **Step 5: Run the Unity Editor entry point with the canonical input**

Launch Unity in batch mode with `-noUpm`, `-projectPath`, `-executeMethod MaaDuelChannel.MapAssembler.DuelMapAssembler.Run`, and `-assemblyConfig <canonical-json-path>`. Inspect Editor logs and the temporary scene hierarchy for all 165 cells and both sets of nine endpoint cells.

### Task 5: Assemble the LMC day environment and camera

**Files:**

- Create: `unity/map-assembler/Assets/Editor/EnvironmentBuilder.cs`
- Create: `unity/map-assembler/Assets/Editor/CameraProfileBuilder.cs`
- Modify: `unity/map-assembler/Assets/Editor/DuelMapAssembler.cs`

**Interfaces:**

- Produces: `EnvironmentBuilder.Build(MapAssemblyInput input, BundleSet bundles) -> RenderLayerRoots` and `CameraProfileBuilder.Build(MapAssemblyInput input, GridBuildResult grid) -> Camera`.
- `RenderLayerRoots` contains separate `GameObject` roots named `environment`, `ground`, `redRoute`, and `blueRoute`.
- Consumes: theme `LMC_DAY`, environment asset records, camera profile, and grid bounds.

- [ ] **Step 1: Instantiate only environment assets resolved for LMC_DAY**

Load the map roots and shared mesh/texture dependencies from `MapAssetPlan`. Create and return separate `RenderLayerRoots` for `environment`, `ground`, `redRoute`, and `blueRoute` so each layer can be rendered independently.

- [ ] **Step 2: Apply original mesh, material, and texture references**

Use original AssetBundle objects and materials. Keep the complete source-to-bundle reference in diagnostics; do not substitute preview PNGs for missing scene assets.

- [ ] **Step 3: Build the stage camera profile**

Apply the profile vectors and `cameraView`, `layerHeight`, and `highlandHeight` values from the stage camera sources. Use grid and environment bounds to check clipping; put any calibrated adjustment in an explicit output camera override, not a hidden code constant.

- [ ] **Step 4: Set a deterministic render state**

Set the selected fixed animation time, disable unneeded particle/random variations, and configure a fixed background clear color and render target dimensions. Do not spawn game units or UI.

### Task 6: Render the clean map, ground image, and semantic masks

**Files:**

- Create: `unity/map-assembler/Assets/Editor/MapImageExporter.cs`
- Create: `unity/map-assembler/Assets/Editor/Shaders/MapMask.shader`
- Modify: `unity/map-assembler/Assets/Editor/DuelMapAssembler.cs`

**Interfaces:**

- Produces: `MapImageExporter.Export(Camera camera, RenderLayerRoots roots, GridBuildResult grid, MapAssemblyInput input) -> MapRenderOutputs`.
- `MapRenderOutputs` paths are `vs2-02a.png`, `ground.png`, `red-route-mask.png`, `blue-route-mask.png`, `assembly-manifest.json`, and `grid-projection.json`.

- [ ] **Step 1: Render the composite map and ground-only layer**

Render the scene at the request width and height. Render `ground.png` with route effects disabled while preserving the base ground tiles and environment.

- [ ] **Step 2: Render independent binary route masks**

Render only route-side geometry to transparent targets using a solid white mask material. Export separate single-channel-equivalent PNG masks for the red and blue side; do not infer masks by color-thresholding the composite output.

- [ ] **Step 3: Export camera-to-grid projection metadata**

Project all 165 `GridBuildResult` cell centers to image coordinates with Unity camera projection. Serialize row, column, tile index, world position, and image pixel position to `grid-projection.json`.

- [ ] **Step 4: Export the assembly manifest and release Unity resources**

Write the stage key, theme, Unity version, input hashes, resolved bundle paths/hashes, tile prefab mapping, camera profile and override, fixed frame time, output size, and output hashes to `assembly-manifest.json`. Release RenderTextures and AssetBundles when the render is complete.

### Task 7: Launch the assembler from Typer and render the first VS-2 image

**Files:**

- Modify: `src/maa_duel/cli.py`
- Modify: `src/maa_duel/map_assembly.py`
- Modify: `src/maa_duel/map_assets.py`
- Create: output files under the selected external workspace only

**Interfaces:**

- Produces: `duel assemble-map` process exit status and the files returned by `MapRenderOutputs`.
- Consumes: CLI options from Task 2, canonical config from Task 2, and Unity Editor entry from Tasks 4–6.

- [ ] **Step 1: Complete process launch and log handling**

Use the explicit Editor path when supplied, otherwise resolve `UNITY_EDITOR` and then the `2021.3.39f1` default install path. Pass `-noUpm` and all user paths as separate subprocess arguments; save Unity output to `output_dir/logs/map-assembly.log`. Return nonzero when Unity exits nonzero or required artifacts are absent.

- [ ] **Step 2: Render VS-2 with local game and ArknightsResource inputs**

Run:

```powershell
uv run duel assemble-map `
  --stage-json 'D:\ArknightsResource-50\gamedata\levels\activities\act1enemyduel\level_act1enemyduel_02a.json' `
  --camera-json 'D:\ArknightsResource-50\map_camera_views\maps.json' `
  --game-assets 'E:\Program Files\Arknights\Arknights_Data' `
  --output-dir 'D:\MAA-DuelChannel\workspace\assets\assembled-maps\vs2-02a'
```

Expected output: all six declared files appear under the output directory; the manifest reports 165 cells, 9 start cells, 9 end cells, `LMC_DAY`, and the loaded bundle hashes.

- [ ] **Step 3: Compare against the game frame and adjust explicit camera configuration**

Open `vs2-02a.png`, both route masks, `ground.png`, and the 1920×864 reference frame side by side. Compare floor grid edges, arena bounds, and red/blue side positions only in the unobstructed battle viewport. If camera adjustment is needed, store it in the map assembly config and regenerate all outputs.

- [ ] **Step 4: Review the final change set**

Run `git status --short` from MaaDuelChannel and confirm the change set contains only source files, Unity project metadata, the asset mapping config, and plan/spec documents. Confirm no raw AssetBundle, rendered output, or Unity-generated directory is tracked.

## Plan Self-Review

- Spec coverage: stage and theme inputs are handled in Tasks 2–3; tile and effect prefabs in Task 4; environment and camera in Task 5; color and semantic outputs in Task 6; CLI, output manifest, hashing, visual comparison, and external output locations in Task 7.
- Error paths: malformed grid/index errors belong to Task 2; missing/hotfixed dependencies to Task 3; missing prefabs to Task 4; camera clipping to Task 5; incomplete or mismatched artifacts to Tasks 6–7.
- No automated test files or test commands are planned; the required user-visible artifact is the map render itself and is checked directly.
