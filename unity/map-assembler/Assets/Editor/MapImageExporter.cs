using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace MaaDuelChannel.MapAssembler
{
    [Serializable]
    public sealed class GridProjectionDocument
    {
        public string stageKey;
        public int imageWidth;
        public int imageHeight;
        public ProjectionCell[] cells;
    }

    [Serializable]
    public sealed class ProjectionCell
    {
        public int row;
        public int column;
        public int tileIndex;
        public string tileKey;
        public float worldX;
        public float worldY;
        public float worldZ;
        public float pixelX;
        public float pixelY;
        public float cameraDepth;
        public bool inFrontOfCamera;
    }

    [Serializable]
    public sealed class AssemblyManifestDocument
    {
        public string stageKey;
        public string theme;
        public string unityVersion;
        public string resourceManifestVersion;
        public string fixedFrameTime;
        public int imageWidth;
        public int imageHeight;
        public int cellCount;
        public int startCellCount;
        public int endCellCount;
        public string gridTransformEvidence;
        public string[] tilePrefabPaths;
        public EnvironmentRootInput[] environmentRoots;
        public ManifestBundle[] bundles;
        public InputHashInput[] inputHashes;
        public CameraManifest camera;
        public string redRouteEffect;
        public string blueRouteEffect;
        public string compositeImage;
        public string groundImage;
        public string redRouteMask;
        public string blueRouteMask;
        public OutputHash[] outputHashes;
    }

    [Serializable]
    public sealed class ManifestBundle
    {
        public string relativePath;
        public string source;
        public string originalPath;
        public string sha256;
        public string normalizedPath;
        public string normalizedSha256;
        public string normalization;
    }

    [Serializable]
    public sealed class CameraManifest
    {
        public float cameraView;
        public float layerHeight;
        public float highlandHeight;
        public float[] cameraOffset;
        public float[] cameraFocus;
        public float[] viewDefault;
        public float[] viewBySide;
        public float[] appliedPosition;
        public float[] appliedFocus;
        public float appliedFieldOfView;
        public string framingRule;
    }

    [Serializable]
    public sealed class OutputHash
    {
        public string file;
        public string sha256;
    }

    public static class MapImageExporter
    {
        public static void Export(
            MapAssemblyInput input,
            GridBuildResult grid,
            RenderLayerRoots roots,
            CameraBuildResult cameraBuild
        )
        {
            string outputDirectory = Path.GetFullPath(input.outputDirectory);
            Directory.CreateDirectory(outputDirectory);
            Camera camera = cameraBuild.Camera;
            camera.targetTexture = null;

            SetRenderRoots(roots, true, true, true, true);
            CaptureAndWrite(camera, outputDirectory, "vs2-02a.png", input.width, input.height, false);

            SetRenderRoots(roots, true, true, false, false);
            CaptureAndWrite(camera, outputDirectory, "ground.png", input.width, input.height, false);

            Material whiteMask = CreateMaskMaterial();
            try
            {
                SetRenderRoots(roots, false, false, true, false);
                CaptureAndWrite(camera, outputDirectory, "red-route-mask.png", input.width, input.height, true, whiteMask);
                SetRenderRoots(roots, false, false, false, true);
                CaptureAndWrite(camera, outputDirectory, "blue-route-mask.png", input.width, input.height, true, whiteMask);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(whiteMask);
            }

            SetRenderRoots(roots, true, true, true, true);
            WriteProjection(input, grid, camera, outputDirectory);
            WriteManifest(input, grid, roots, cameraBuild, outputDirectory);
            Debug.Log("Exported VS-2 map images, independent route masks, projection, and asset manifest to " + outputDirectory);
        }

        private static void SetRenderRoots(
            RenderLayerRoots roots,
            bool environment,
            bool ground,
            bool redRoute,
            bool blueRoute
        )
        {
            SetActive(roots.Environment, environment);
            SetActive(roots.Ground, ground);
            SetActive(roots.RedRoute, redRoute);
            SetActive(roots.BlueRoute, blueRoute);
        }

        private static void SetActive(GameObject root, bool active)
        {
            if (root != null)
            {
                root.SetActive(active);
            }
        }

        private static void CaptureAndWrite(
            Camera camera,
            string outputDirectory,
            string fileName,
            int width,
            int height,
            bool binaryMask,
            Material replacementMaterial = null
        )
        {
            List<RendererMaterialState> originalMaterials = replacementMaterial == null
                ? null
                : ReplaceAllVisibleMaterials(replacementMaterial);
            RenderTexture target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32)
            {
                antiAliasing = 1,
                name = "MapRender_" + fileName,
            };
            RenderTexture previousTarget = camera.targetTexture;
            RenderTexture previousActive = RenderTexture.active;
            try
            {
                camera.targetTexture = target;
                camera.Render();
                RenderTexture.active = target;
                Texture2D image = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
                image.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                image.Apply(false, false);
                if (binaryMask)
                {
                    MakeBinaryMask(image);
                }

                byte[] png = image.EncodeToPNG();
                File.WriteAllBytes(Path.Combine(outputDirectory, fileName), png);
                UnityEngine.Object.DestroyImmediate(image);
            }
            finally
            {
                camera.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                target.Release();
                UnityEngine.Object.DestroyImmediate(target);
                if (originalMaterials != null)
                {
                    RestoreMaterials(originalMaterials);
                }
            }
        }

        private static List<RendererMaterialState> ReplaceAllVisibleMaterials(Material replacement)
        {
            Renderer[] renderers = UnityEngine.Object.FindObjectsOfType<Renderer>();
            List<RendererMaterialState> state = new List<RendererMaterialState>(renderers.Length);
            foreach (Renderer renderer in renderers)
            {
                if (renderer == null || !renderer.gameObject.activeInHierarchy)
                {
                    continue;
                }

                Material[] original = renderer.sharedMaterials;
                Material[] maskMaterials = new Material[original.Length];
                for (int index = 0; index < maskMaterials.Length; index++)
                {
                    maskMaterials[index] = replacement;
                }

                state.Add(new RendererMaterialState { Renderer = renderer, Materials = original });
                renderer.sharedMaterials = maskMaterials;
            }

            return state;
        }

        private static void RestoreMaterials(List<RendererMaterialState> state)
        {
            foreach (RendererMaterialState item in state)
            {
                if (item.Renderer != null)
                {
                    item.Renderer.sharedMaterials = item.Materials;
                }
            }
        }

        private static Material CreateMaskMaterial()
        {
            Shader shader = Shader.Find("Unlit/Color");
            if (shader == null)
            {
                shader = Shader.Find("Sprites/Default");
            }

            if (shader == null)
            {
                throw new InvalidDataException("Unity could not find a built-in unlit shader for route masks.");
            }

            return new Material(shader) { color = Color.white, name = "DuelRouteMask_White" };
        }

        private static void MakeBinaryMask(Texture2D image)
        {
            Color32[] pixels = image.GetPixels32();
            for (int index = 0; index < pixels.Length; index++)
            {
                Color32 pixel = pixels[index];
                bool covered = pixel.a > 8 && (pixel.r > 8 || pixel.g > 8 || pixel.b > 8);
                pixels[index] = covered ? new Color32(255, 255, 255, 255) : new Color32(0, 0, 0, 0);
            }

            image.SetPixels32(pixels);
            image.Apply(false, false);
        }

        private static void WriteProjection(MapAssemblyInput input, GridBuildResult grid, Camera camera, string outputDirectory)
        {
            RenderTexture projectionTarget = new RenderTexture(input.width, input.height, 0, RenderTextureFormat.ARGB32)
            {
                name = "MapProjectionTarget",
            };
            RenderTexture previousTarget = camera.targetTexture;
            ProjectionCell[] projected = new ProjectionCell[grid.Cells.Count];
            try
            {
                camera.targetTexture = projectionTarget;
                for (int index = 0; index < grid.Cells.Count; index++)
                {
                    GridCellRecord cell = grid.Cells[index];
                    Vector3 world = cell.WorldPosition;
                    Vector3 screen = camera.WorldToScreenPoint(world);
                    projected[index] = new ProjectionCell
                    {
                        row = cell.Cell.row,
                        column = cell.Cell.column,
                        tileIndex = cell.Cell.tileIndex,
                        tileKey = cell.Cell.tileKey,
                        worldX = world.x,
                        worldY = world.y,
                        worldZ = world.z,
                        pixelX = screen.x,
                        pixelY = input.height - screen.y,
                        cameraDepth = screen.z,
                        inFrontOfCamera = screen.z > 0f,
                    };
                }
            }
            finally
            {
                camera.targetTexture = previousTarget;
                projectionTarget.Release();
                UnityEngine.Object.DestroyImmediate(projectionTarget);
            }

            GridProjectionDocument document = new GridProjectionDocument
            {
                stageKey = input.stageKey,
                imageWidth = input.width,
                imageHeight = input.height,
                cells = projected,
            };
            File.WriteAllText(
                Path.Combine(outputDirectory, "grid-projection.json"),
                JsonUtility.ToJson(document, true) + "\n",
                new UTF8Encoding(false)
            );
        }

        private static void WriteManifest(
            MapAssemblyInput input,
            GridBuildResult grid,
            RenderLayerRoots roots,
            CameraBuildResult cameraBuild,
            string outputDirectory
        )
        {
            List<ManifestBundle> bundles = new List<ManifestBundle>(input.bundleFiles.Length);
            foreach (BundleFileInput bundle in input.bundleFiles)
            {
                bundles.Add(new ManifestBundle
                {
                    relativePath = bundle.relativePath,
                    source = bundle.source,
                    originalPath = bundle.originalPath,
                    sha256 = bundle.sha256,
                    normalizedPath = bundle.path,
                    normalizedSha256 = bundle.normalizedSha256,
                    normalization = bundle.normalization,
                });
            }

            int startCount = 0;
            int endCount = 0;
            foreach (GridCellRecord cell in grid.Cells)
            {
                if (cell.Cell.tileKey == "tile_start_dqq")
                {
                    startCount++;
                }
                else if (cell.Cell.tileKey == "tile_end_dqq")
                {
                    endCount++;
                }
            }

            string[] tilePrefabPaths = new string[input.tilePrefabByKey.Length];
            for (int index = 0; index < tilePrefabPaths.Length; index++)
            {
                TilePrefabInput tile = input.tilePrefabByKey[index];
                tilePrefabPaths[index] = tile.tileKey + "=" + tile.bundlePath + ":" + tile.assetPath;
            }

            string[] outputNames = { "vs2-02a.png", "ground.png", "red-route-mask.png", "blue-route-mask.png", "grid-projection.json" };
            OutputHash[] outputHashes = new OutputHash[outputNames.Length];
            for (int index = 0; index < outputNames.Length; index++)
            {
                string filePath = Path.Combine(outputDirectory, outputNames[index]);
                outputHashes[index] = new OutputHash { file = outputNames[index], sha256 = Sha256(File.ReadAllBytes(filePath)) };
            }

            CameraProfileInput profile = input.camera;
            AssemblyManifestDocument document = new AssemblyManifestDocument
            {
                stageKey = input.stageKey,
                theme = input.theme,
                unityVersion = Application.unityVersion,
                resourceManifestVersion = input.resourceManifestVersion,
                fixedFrameTime = input.frameTime.ToString("F3", System.Globalization.CultureInfo.InvariantCulture),
                imageWidth = input.width,
                imageHeight = input.height,
                cellCount = grid.Cells.Count,
                startCellCount = startCount,
                endCellCount = endCount,
                gridTransformEvidence = input.worldGrid.evidence,
                tilePrefabPaths = tilePrefabPaths,
                environmentRoots = input.environmentRoots,
                bundles = bundles.ToArray(),
                inputHashes = input.inputHashes,
                camera = new CameraManifest
                {
                    cameraView = profile.cameraView,
                    layerHeight = profile.layerHeight,
                    highlandHeight = profile.highlandHeight,
                    cameraOffset = profile.cameraOffset,
                    cameraFocus = profile.cameraFocus,
                    viewDefault = profile.viewDefault,
                    viewBySide = profile.viewBySide,
                    appliedPosition = VectorArray(cameraBuild.Position),
                    appliedFocus = VectorArray(cameraBuild.Focus),
                    appliedFieldOfView = cameraBuild.FieldOfView,
                    framingRule = "Perspective; exact viewDefault position and cameraFocus target; vertical grid span plus cameraView margin.",
                },
                redRouteEffect = roots.RedRouteEffect,
                blueRouteEffect = roots.BlueRouteEffect,
                compositeImage = "vs2-02a.png",
                groundImage = "ground.png",
                redRouteMask = "red-route-mask.png",
                blueRouteMask = "blue-route-mask.png",
                outputHashes = outputHashes,
            };
            File.WriteAllText(
                Path.Combine(outputDirectory, "assembly-manifest.json"),
                JsonUtility.ToJson(document, true) + "\n",
                new UTF8Encoding(false)
            );
        }

        private static float[] VectorArray(Vector3 value)
        {
            return new[] { value.x, value.y, value.z };
        }

        private static string Sha256(byte[] contents)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(contents);
                StringBuilder result = new StringBuilder(digest.Length * 2);
                foreach (byte value in digest)
                {
                    result.Append(value.ToString("x2"));
                }

                return result.ToString();
            }
        }

        private sealed class RendererMaterialState
        {
            public Renderer Renderer;
            public Material[] Materials;
        }
    }
}
