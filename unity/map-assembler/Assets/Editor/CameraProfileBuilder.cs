using System;
using System.IO;
using UnityEngine;

namespace MaaDuelChannel.MapAssembler
{
    public sealed class CameraBuildResult
    {
        public Camera Camera;
        public Vector3 Position;
        public Vector3 Focus;
        public float FieldOfView;
    }

    public static class CameraProfileBuilder
    {
        public static CameraBuildResult Build(MapAssemblyInput input, GridBuildResult grid)
        {
            CameraProfileInput profile = input.camera;
            Vector3 position = ToVector3(profile.viewDefault, "viewDefault");
            Vector3 focus = ToVector3(profile.cameraFocus, "cameraFocus");
            float distance = Vector3.Distance(position, focus);
            if (distance <= 0.01f)
            {
                throw new InvalidDataException("Camera position and focus must be different.");
            }

            float verticalSpan = grid.WorldBounds.size.y + Mathf.Max(0f, profile.cameraView);
            float halfAngle = Mathf.Atan(verticalSpan / (2f * distance)) * Mathf.Rad2Deg;
            float fieldOfView = Mathf.Clamp(halfAngle * 2f, 10f, 100f);
            GameObject cameraObject = new GameObject("VS2_Map_Camera");
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.transform.position = position;
            camera.transform.LookAt(focus, Vector3.up);
            camera.orthographic = false;
            camera.fieldOfView = fieldOfView;
            camera.aspect = input.width / (float)input.height;
            camera.nearClipPlane = 0.01f;
            camera.farClipPlane = 500f;
            camera.allowHDR = false;
            camera.allowMSAA = true;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0f, 0f, 0f, 0f);

            Debug.Log(
                "Built map camera position=" + position + ", focus=" + focus + ", distance=" + distance.ToString("F3")
                + ", FOV=" + fieldOfView.ToString("F3") + "deg, cameraView padding=" + profile.cameraView.ToString("F3")
            );
            return new CameraBuildResult
            {
                Camera = camera,
                Position = position,
                Focus = focus,
                FieldOfView = fieldOfView,
            };
        }

        private static Vector3 ToVector3(float[] values, string label)
        {
            if (values == null || values.Length != 3)
            {
                throw new InvalidDataException("Camera " + label + " must have three components.");
            }

            return new Vector3(values[0], values[1], values[2]);
        }
    }
}
