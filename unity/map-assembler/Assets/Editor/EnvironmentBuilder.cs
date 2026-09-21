using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace MaaDuelChannel.MapAssembler
{
    public sealed class RenderLayerRoots
    {
        public GameObject Environment;
        public GameObject Ground;
        public GameObject RedRoute;
        public GameObject BlueRoute;
        public string RedRouteEffect;
        public string BlueRouteEffect;
    }

    public static class EnvironmentBuilder
    {
        public static RenderLayerRoots Build(MapAssemblyInput input, BundleSet bundles, GridBuildResult grid)
        {
            GameObject environment = new GameObject("environment");
            GameObject ground = new GameObject("ground");
            GameObject redRoute = new GameObject("red-route");
            GameObject blueRoute = new GameObject("blue-route");
            ground.SetActive(false);
            redRoute.SetActive(false);
            blueRoute.SetActive(false);
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.72f, 0.72f, 0.72f, 1f);
            RenderSettings.ambientIntensity = 1f;

            foreach (EnvironmentRootInput mapping in input.environmentRoots)
            {
                GameObject prefab = bundles.LoadGameObject(mapping.bundlePath, mapping.assetPath);
                Transform parent = string.Equals(mapping.layer, "ground", StringComparison.OrdinalIgnoreCase)
                    ? ground.transform
                    : environment.transform;
                GameObject instance = UnityEngine.Object.Instantiate(
                    prefab,
                    prefab.transform.position,
                    prefab.transform.rotation,
                    parent
                );
                instance.name = mapping.name;
                instance.transform.localScale = prefab.transform.localScale;
                ApplyMaterialOverride(mapping, bundles, instance);
                DisableNondeterministicSystems(instance);
                LogRendererMaterials(mapping.name, instance);
            }

            GameObject lightObject = new GameObject("map-assembly-key-light");
            lightObject.transform.SetParent(environment.transform, false);
            lightObject.transform.rotation = Quaternion.LookRotation(Vector3.forward, Vector3.up);
            Light keyLight = lightObject.AddComponent<Light>();
            keyLight.type = LightType.Directional;
            keyLight.color = Color.white;
            keyLight.intensity = 1.1f;
            keyLight.shadows = LightShadows.None;

            RouteEffectInput startEffect = FindRouteEffect(input.routeEffects, "tile_start");
            RouteEffectInput endEffect = FindRouteEffect(input.routeEffects, "tile_end_multiplayer_05");
            int redCount = 0;
            int blueCount = 0;
            foreach (GridCellRecord cell in grid.Cells)
            {
                if (cell.Cell.tileKey == "tile_end_dqq")
                {
                    InstantiateRouteEffect(bundles, endEffect, redRoute.transform, cell.WorldPosition, "RedEnd", redCount++);
                }
                else if (cell.Cell.tileKey == "tile_start_dqq")
                {
                    InstantiateRouteEffect(bundles, startEffect, blueRoute.transform, cell.WorldPosition, "BlueStart", blueCount++);
                }
            }

            if (redCount != 9 || blueCount != 9)
            {
                throw new InvalidDataException("VS-2 route markers must resolve to exactly nine cells per side.");
            }

            Debug.Log(
                "Instantiated environment roots=" + input.environmentRoots.Length + ", red markers=" + redCount
                + " (" + endEffect.assetPath + "), blue markers=" + blueCount + " (" + startEffect.assetPath + ")."
            );
            return new RenderLayerRoots
            {
                Environment = environment,
                Ground = ground,
                RedRoute = redRoute,
                BlueRoute = blueRoute,
                RedRouteEffect = endEffect.assetPath,
                BlueRouteEffect = startEffect.assetPath,
            };
        }

        private static void LogRendererMaterials(string label, GameObject root)
        {
            Renderer[] renderers = root.GetComponentsInChildren<Renderer>(true);
            foreach (Renderer renderer in renderers)
            {
                foreach (Material material in renderer.sharedMaterials)
                {
                    string shaderName = material == null || material.shader == null ? "<missing>" : material.shader.name;
                    Debug.Log(
                        "Render resource " + label + ": renderer=" + renderer.name + ", material="
                        + (material == null ? "<missing>" : material.name) + ", shader=" + shaderName
                    );
                }
            }
        }

        private static void ApplyMaterialOverride(EnvironmentRootInput mapping, BundleSet bundles, GameObject instance)
        {
            if (string.IsNullOrWhiteSpace(mapping.materialBundlePath) && string.IsNullOrWhiteSpace(mapping.materialAssetPath))
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(mapping.materialBundlePath) || string.IsNullOrWhiteSpace(mapping.materialAssetPath))
            {
                throw new InvalidDataException("Environment material override must specify both bundle and asset path: " + mapping.name);
            }

            Material material = bundles.LoadMaterial(mapping.materialBundlePath, mapping.materialAssetPath);
            foreach (Renderer renderer in instance.GetComponentsInChildren<Renderer>(true))
            {
                int slotCount = Mathf.Max(1, renderer.sharedMaterials.Length);
                Material[] materials = new Material[slotCount];
                for (int index = 0; index < materials.Length; index++)
                {
                    materials[index] = material;
                }

                renderer.sharedMaterials = materials;
            }
        }

        private static RouteEffectInput FindRouteEffect(RouteEffectInput[] effects, string name)
        {
            foreach (RouteEffectInput effect in effects)
            {
                if (effect != null && string.Equals(effect.name, name, StringComparison.Ordinal))
                {
                    return effect;
                }
            }

            throw new InvalidDataException("Required route effect is not declared in the assembly input: " + name);
        }

        private static void InstantiateRouteEffect(
            BundleSet bundles,
            RouteEffectInput effect,
            Transform parent,
            Vector3 position,
            string label,
            int index
        )
        {
            GameObject prefab = bundles.LoadGameObject(effect.bundlePath, effect.assetPath);
            GameObject instance = UnityEngine.Object.Instantiate(prefab, position, prefab.transform.rotation, parent);
            instance.name = label + "_" + index.ToString("D2");
            instance.transform.localScale = prefab.transform.localScale;
            DisableNondeterministicSystems(instance);
            if (index == 0)
            {
                Bounds bounds = default;
                bool hasBounds = false;
                Renderer[] renderers = instance.GetComponentsInChildren<Renderer>(true);
                foreach (Renderer renderer in renderers)
                {
                    if (renderer is ParticleSystemRenderer)
                    {
                        continue;
                    }

                    Debug.Log(
                        "Route mesh " + effect.name + ": " + renderer.name + " bounds=" + renderer.bounds
                        + ", root=" + instance.transform.position
                    );
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

                Debug.Log(
                    "Route marker " + effect.name + " instance=" + instance.transform.position + ", rendererCount="
                    + renderers.Length + ", bounds=" + (hasBounds ? bounds.ToString() : "<no renderer>")
                );
            }
        }

        private static void DisableNondeterministicSystems(GameObject root)
        {
            foreach (ParticleSystem system in root.GetComponentsInChildren<ParticleSystem>(true))
            {
                system.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
                system.Simulate(0f, true, true, true);
            }

            foreach (Animator animator in root.GetComponentsInChildren<Animator>(true))
            {
                animator.enabled = false;
            }
        }
    }
}
