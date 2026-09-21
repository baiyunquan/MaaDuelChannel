using System;
using System.IO;
using UnityEngine;

namespace MaaDuelChannel.MapAssembler
{
    [Serializable]
    public sealed class MapAssemblyInput
    {
        public string stageKey;
        public string theme;
        public int width;
        public int height;
        public float frameTime;
        public StageCellInput[] grid;
        public CameraProfileInput camera;
        public WorldGridInput worldGrid;
        public string bundleRoot;
        public BundleFileInput[] bundleFiles;
        public string outputDirectory;
        public InputHashInput[] inputHashes;
        public TilePrefabInput[] tilePrefabByKey;
        public RouteEffectInput[] routeEffects;
        public EnvironmentRootInput[] environmentRoots;
        public string resourceManifestVersion;
        public BundleCompatibilityInput bundleCompatibility;
    }

    [Serializable]
    public sealed class WorldGridInput
    {
        public float[] origin;
        public int columnDirection;
        public int rowDirection;
        public float columnPitch;
        public float rowPitch;
        public string evidence;
    }

    [Serializable]
    public sealed class StageCellInput
    {
        public int row;
        public int column;
        public int tileIndex;
        public string tileKey;
        public int heightType;
        public int buildableType;
        public int passableMask;
        public int playerSideMask;
    }

    [Serializable]
    public sealed class CameraProfileInput
    {
        public CameraViewsInput profile;
        public float cameraView;
        public float layerHeight;
        public float highlandHeight;
        public float[] cameraOffset;
        public float[] cameraFocus;
        public float[] viewDefault;
        public float[] viewBySide;
    }

    [Serializable]
    public sealed class CameraViewsInput
    {
        public float[] defaultView;
        public float[] sideView;
    }

    [Serializable]
    public sealed class BundleFileInput
    {
        public string path;
        public string relativePath;
        public string sha256;
        public string source;
        public int manifestIndex;
        public string originalPath;
        public string normalizedSha256;
        public string normalization;
    }

    [Serializable]
    public sealed class BundleCompatibilityInput
    {
        public string normalization;
        public string unityPyVersion;
        public string arkUnpackerCommit;
    }

    [Serializable]
    public sealed class InputHashInput
    {
        public string role;
        public string path;
        public string sha256;
    }

    [Serializable]
    public sealed class TilePrefabInput
    {
        public string tileKey;
        public string bundlePath;
        public string assetPath;
    }

    [Serializable]
    public sealed class RouteEffectInput
    {
        public string name;
        public string bundlePath;
        public string assetPath;
    }

    [Serializable]
    public sealed class EnvironmentRootInput
    {
        public string name;
        public string bundlePath;
        public string assetPath;
        public string layer;
        public string materialBundlePath;
        public string materialAssetPath;
    }

    public static class AssemblyInput
    {
        public static MapAssemblyInput Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException("Assembly input path must not be empty.", nameof(path));
            }

            string json = File.ReadAllText(path);
            MapAssemblyInput input = JsonUtility.FromJson<MapAssemblyInput>(json);
            if (input == null || string.IsNullOrWhiteSpace(input.stageKey) || string.IsNullOrWhiteSpace(input.theme))
            {
                throw new InvalidDataException("Assembly input is missing stageKey or theme.");
            }

            if (input.width <= 0 || input.height <= 0 || input.grid == null || input.grid.Length == 0)
            {
                throw new InvalidDataException("Assembly input must have a positive output size and a non-empty grid.");
            }

            if (input.camera == null || input.camera.profile == null)
            {
                throw new InvalidDataException("Assembly input is missing the camera profile.");
            }

            if (input.worldGrid == null || input.worldGrid.origin == null || input.worldGrid.origin.Length != 3
                || input.worldGrid.columnPitch <= 0f || input.worldGrid.rowPitch <= 0f
                || Math.Abs(input.worldGrid.columnDirection) != 1 || Math.Abs(input.worldGrid.rowDirection) != 1)
            {
                throw new InvalidDataException("Assembly input is missing a valid world-grid transform.");
            }

            input.bundleFiles = input.bundleFiles ?? Array.Empty<BundleFileInput>();
            input.inputHashes = input.inputHashes ?? Array.Empty<InputHashInput>();
            input.tilePrefabByKey = input.tilePrefabByKey ?? Array.Empty<TilePrefabInput>();
            input.routeEffects = input.routeEffects ?? Array.Empty<RouteEffectInput>();
            input.environmentRoots = input.environmentRoots ?? Array.Empty<EnvironmentRootInput>();
            return input;
        }
    }
}
