using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace MaaDuelChannel.MapAssembler
{
    public static class DuelMapAssembler
    {
        public static void Run()
        {
            try
            {
                string configPath = GetAssemblyConfigPath();
                MapAssemblyInput input = AssemblyInput.Load(configPath);
                using (BundleSet bundles = BundleSet.Load(input))
                {
                    GridBuildResult grid = null;
                    RenderLayerRoots roots = null;
                    CameraBuildResult camera = null;
                    try
                    {
                        grid = StageGridBuilder.Build(input, bundles);
                        roots = EnvironmentBuilder.Build(input, bundles, grid);
                        camera = CameraProfileBuilder.Build(input, grid);
                        MapImageExporter.Export(input, grid, roots, camera);
                        Debug.Log(
                            "Assembly input accepted: stage=" + input.stageKey + ", theme=" + input.theme
                            + ", cells=" + grid.Cells.Count + ", bundles=" + bundles.Count + ", config=" + Path.GetFullPath(configPath)
                        );
                    }
                    finally
                    {
                        if (camera != null && camera.Camera != null)
                        {
                            UnityEngine.Object.DestroyImmediate(camera.Camera.gameObject);
                        }

                        if (roots != null)
                        {
                            DestroyRoot(roots.Environment);
                            DestroyRoot(roots.Ground);
                            DestroyRoot(roots.RedRoute);
                            DestroyRoot(roots.BlueRoute);
                        }

                        if (grid != null)
                        {
                            DestroyRoot(grid.Root);
                        }
                    }
                }
            }
            catch (Exception error)
            {
                Debug.LogException(error);
                EditorApplication.Exit(1);
            }
        }

        private static void DestroyRoot(GameObject root)
        {
            if (root != null)
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static string GetAssemblyConfigPath()
        {
            string[] arguments = Environment.GetCommandLineArgs();
            for (int index = 0; index < arguments.Length - 1; index++)
            {
                if (string.Equals(arguments[index], "-assemblyConfig", StringComparison.OrdinalIgnoreCase))
                {
                    return arguments[index + 1];
                }
            }

            throw new InvalidDataException("Missing -assemblyConfig <path> command-line argument.");
        }
    }
}
