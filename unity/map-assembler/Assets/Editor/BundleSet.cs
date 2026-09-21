using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace MaaDuelChannel.MapAssembler
{
    public sealed class BundleSet : IDisposable
    {
        private readonly Dictionary<string, AssetBundle> _bundles = new Dictionary<string, AssetBundle>(StringComparer.OrdinalIgnoreCase);
        private readonly List<AssetBundle> _loadOrder = new List<AssetBundle>();
        private bool _disposed;

        private BundleSet()
        {
        }

        public int Count => _loadOrder.Count;

        public static BundleSet Load(MapAssemblyInput input)
        {
            if (input == null || input.bundleFiles == null || input.bundleFiles.Length == 0)
            {
                throw new InvalidDataException("Assembly input has no resolved AssetBundle files.");
            }

            BundleSet result = new BundleSet();
            try
            {
                foreach (BundleFileInput file in input.bundleFiles)
                {
                    if (file == null || string.IsNullOrWhiteSpace(file.path) || string.IsNullOrWhiteSpace(file.relativePath))
                    {
                        throw new InvalidDataException("Assembly input contains an invalid bundle file record.");
                    }

                    if (!File.Exists(file.path))
                    {
                        throw new FileNotFoundException("Resolved AssetBundle file does not exist.", file.path);
                    }

                    if (result._bundles.ContainsKey(file.relativePath))
                    {
                        throw new InvalidDataException("AssetBundle list contains a duplicate path: " + file.relativePath);
                    }

                    AssetBundle bundle = AssetBundle.LoadFromFile(file.path);
                    if (bundle == null)
                    {
                        throw new InvalidDataException(
                            "Unity could not load AssetBundle '" + file.relativePath + "' from '" + file.path + "'."
                        );
                    }

                    result._bundles.Add(file.relativePath, bundle);
                    result._loadOrder.Add(bundle);
                }

                Debug.Log("Loaded " + result.Count + " AssetBundles in resolved dependency order.");
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        public AssetBundle GetBundle(string relativePath)
        {
            ThrowIfDisposed();
            if (string.IsNullOrWhiteSpace(relativePath) || !_bundles.TryGetValue(relativePath, out AssetBundle bundle))
            {
                throw new KeyNotFoundException("AssetBundle is not loaded: " + relativePath);
            }

            return bundle;
        }

        public GameObject LoadGameObject(string relativeBundlePath, string assetPath)
        {
            if (string.IsNullOrWhiteSpace(assetPath))
            {
                throw new ArgumentException("Asset path must not be empty.", nameof(assetPath));
            }

            AssetBundle bundle = GetBundle(relativeBundlePath);
            GameObject prefab = bundle.LoadAsset<GameObject>(assetPath);
            if (prefab == null)
            {
                string[] available = bundle.GetAllAssetNames();
                string closest = FindPathHint(available, assetPath);
                throw new InvalidDataException(
                    "AssetBundle '" + relativeBundlePath + "' has no GameObject at '" + assetPath + "'." + closest
                );
            }

            return prefab;
        }

        public Material LoadMaterial(string relativeBundlePath, string assetPath)
        {
            if (string.IsNullOrWhiteSpace(assetPath))
            {
                throw new ArgumentException("Material asset path must not be empty.", nameof(assetPath));
            }

            AssetBundle bundle = GetBundle(relativeBundlePath);
            Material material = bundle.LoadAsset<Material>(assetPath);
            if (material == null)
            {
                string[] available = bundle.GetAllAssetNames();
                string closest = FindPathHint(available, assetPath);
                throw new InvalidDataException(
                    "AssetBundle '" + relativeBundlePath + "' has no Material at '" + assetPath + "'." + closest
                );
            }

            return material;
        }

        public string DescribeGameObject(string relativeBundlePath, string assetPath)
        {
            GameObject prefab = LoadGameObject(relativeBundlePath, assetPath);
            Renderer[] renderers = prefab.GetComponentsInChildren<Renderer>(true);
            bool hasBounds = false;
            Bounds bounds = new Bounds(prefab.transform.position, Vector3.zero);
            foreach (Renderer renderer in renderers)
            {
                if (renderer == null)
                {
                    continue;
                }

                if (!hasBounds)
                {
                    bounds = renderer.bounds;
                    hasBounds = true;
                }
                else
                {
                    bounds.Encapsulate(renderer.bounds);
                }
            }

            string boundsText = hasBounds
                ? "boundsSize=" + FormatVector(bounds.size) + " boundsCenter=" + FormatVector(bounds.center)
                : "boundsSize=<no renderer>";
            return prefab.name + " childCount=" + prefab.transform.childCount + " rendererCount=" + renderers.Length + " " + boundsText;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            for (int index = _loadOrder.Count - 1; index >= 0; index--)
            {
                AssetBundle bundle = _loadOrder[index];
                if (bundle != null)
                {
                    bundle.Unload(false);
                }
            }

            _loadOrder.Clear();
            _bundles.Clear();
            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(BundleSet));
            }
        }

        private static string FindPathHint(string[] candidates, string requestedPath)
        {
            string requestedName = Path.GetFileNameWithoutExtension(requestedPath);
            foreach (string candidate in candidates)
            {
                if (string.Equals(Path.GetFileNameWithoutExtension(candidate), requestedName, StringComparison.OrdinalIgnoreCase))
                {
                    return " Similar bundled path: '" + candidate + "'.";
                }
            }

            return candidates.Length == 0 ? " The bundle has no named assets." : " Bundle asset count: " + candidates.Length + ".";
        }

        private static string FormatVector(Vector3 value)
        {
            return "(" + value.x.ToString("F4") + ", " + value.y.ToString("F4") + ", " + value.z.ToString("F4") + ")";
        }
    }
}
