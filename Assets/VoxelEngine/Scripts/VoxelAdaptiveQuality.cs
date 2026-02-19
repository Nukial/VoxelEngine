using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Dynamically adjusts ray-march quality based on camera motion, GPU load, 
    /// distance to volume, and edge proximity.
    /// </summary>
    public class VoxelAdaptiveQuality
    {
        // Output: computed per frame
        public int RuntimeMaxRaySteps { get; private set; }
        public int RuntimeMaxShadowSteps { get; private set; }
        public float MotionBlend { get; private set; }
        public float GpuLoadBlend { get; private set; }
        public bool EnableGpuAdaptive { get; private set; }

        // Internal state
        private Vector3 _lastCameraPos;
        private Quaternion _lastCameraRot;
        private bool _cameraPosInitialized;

        /// <summary>
        /// Recompute quality parameters for the current frame.
        /// Call once per frame before UpdateRenderProperties.
        /// </summary>
        public void Update(
            int maxRaySteps, int maxShadowSteps,
            bool enableAdaptiveQuality, int movingRaySteps, int movingShadowSteps,
            float movingShadowStepFloor, float cameraMotionThreshold, float qualityTransitionSpeed,
            bool enableGpuAdaptiveQuality, float gpuTargetFrameRate,
            float gpuRayStepFloor, float gpuShadowStepFloor,
            bool enableEdgeLoadGuard, float edgeLoadDistance,
            float edgeRayStepScale, float edgeShadowStepScale,
            float maxRenderDistance, float distanceQualityFactor, float renderDistanceFadeRatio,
            Vector3 worldOrigin, float worldExtent)
        {
            EnableGpuAdaptive = enableGpuAdaptiveQuality;

            // --- Motion blend ---
            float rawMotion = GetCameraMotionIntensity(cameraMotionThreshold);
            float targetBlend = rawMotion > 0f ? Mathf.Clamp01(rawMotion) : 0f;
            MotionBlend = Mathf.Lerp(MotionBlend, targetBlend, Time.deltaTime * qualityTransitionSpeed);
            if (MotionBlend < 0.01f) MotionBlend = 0f;

            int runtimeRay = maxRaySteps;
            int runtimeShadow = maxShadowSteps;

            // --- GPU load blend ---
            if (enableGpuAdaptiveQuality)
            {
                float targetFrameTime = 1f / Mathf.Max(1f, gpuTargetFrameRate);
                float smoothedFrame = Mathf.Max(Time.smoothDeltaTime, Time.deltaTime);
                float gpuPressure = Mathf.Clamp01((smoothedFrame - targetFrameTime) /
                    Mathf.Max(targetFrameTime * 0.8f, 0.0001f));
                GpuLoadBlend = Mathf.Lerp(GpuLoadBlend, gpuPressure,
                    Time.deltaTime * qualityTransitionSpeed * 0.7f);
            }
            else
            {
                GpuLoadBlend = 0f;
            }

            // --- Adaptive quality (motion) ---
            if (enableAdaptiveQuality && MotionBlend > 0f)
            {
                runtimeRay = Mathf.RoundToInt(
                    Mathf.Lerp(maxRaySteps, Mathf.Min(maxRaySteps, movingRaySteps), MotionBlend));

                int movingShadowTarget = Mathf.Min(maxShadowSteps, movingShadowSteps);
                int shadowFloor = Mathf.RoundToInt(maxShadowSteps * movingShadowStepFloor);
                movingShadowTarget = Mathf.Max(movingShadowTarget, shadowFloor);
                runtimeShadow = Mathf.RoundToInt(
                    Mathf.Lerp(maxShadowSteps, movingShadowTarget, MotionBlend));
            }

            // --- Adaptive quality (GPU load) ---
            if (enableGpuAdaptiveQuality && GpuLoadBlend > 0f)
            {
                int gpuRayTarget = Mathf.Max(64, Mathf.RoundToInt(maxRaySteps * gpuRayStepFloor));
                int gpuShadowTarget = Mathf.Max(8, Mathf.RoundToInt(maxShadowSteps * gpuShadowStepFloor));
                runtimeRay = Mathf.RoundToInt(Mathf.Lerp(runtimeRay, gpuRayTarget, GpuLoadBlend));
                runtimeShadow = Mathf.RoundToInt(Mathf.Lerp(runtimeShadow, gpuShadowTarget, GpuLoadBlend));
            }

            // --- Distance-based quality ---
            var camera = Camera.main;
            if (camera != null)
            {
                float distToCenter = Vector3.Distance(camera.transform.position,
                    worldOrigin + Vector3.one * worldExtent * 0.5f);

                float effectiveMaxDist = Mathf.Max(1f, maxRenderDistance);
                float fadeStart = effectiveMaxDist * Mathf.Clamp01(renderDistanceFadeRatio);
                float fadeRange = Mathf.Max(0.001f, effectiveMaxDist - fadeStart);
                float distRatio = Mathf.Clamp01((distToCenter - fadeStart) / fadeRange);

                float qualityMult = Mathf.Lerp(1f, distanceQualityFactor, distRatio);
                runtimeRay = Mathf.Max(64, Mathf.RoundToInt(runtimeRay * qualityMult));
                runtimeShadow = Mathf.Max(16, Mathf.RoundToInt(runtimeShadow * qualityMult));

                // --- Edge load guard ---
                if (enableEdgeLoadGuard)
                {
                    bool insideVolume = IsCameraInsideVolume(worldOrigin, worldExtent, 0f);
                    if (!insideVolume)
                    {
                        float distanceToBounds = GetDistanceToVolumeBounds(
                            camera.transform.position, worldOrigin, worldExtent);
                        float edgeFactor = 1f - Mathf.Clamp01(
                            distanceToBounds / Mathf.Max(0.1f, edgeLoadDistance));

                        if (edgeFactor > 0f)
                        {
                            float rayScale = Mathf.Lerp(1f, edgeRayStepScale, edgeFactor);
                            float shadowScale = Mathf.Lerp(1f, edgeShadowStepScale, edgeFactor);
                            runtimeRay = Mathf.Max(64, Mathf.RoundToInt(runtimeRay * rayScale));
                            runtimeShadow = Mathf.Max(16, Mathf.RoundToInt(runtimeShadow * shadowScale));
                        }
                    }
                }
            }

            RuntimeMaxRaySteps = runtimeRay;
            RuntimeMaxShadowSteps = runtimeShadow;
        }

        // ----- Helpers -----

        private float GetCameraMotionIntensity(float cameraMotionThreshold)
        {
            var camera = Camera.main;
            if (camera == null) return 0f;

            Vector3 currentPos = camera.transform.position;
            Quaternion currentRot = camera.transform.rotation;

            if (!_cameraPosInitialized)
            {
                _lastCameraPos = currentPos;
                _lastCameraRot = currentRot;
                _cameraPosInitialized = true;
                return 0f;
            }

            float posDelta = (currentPos - _lastCameraPos).magnitude;
            float rotDelta = Quaternion.Angle(_lastCameraRot, currentRot);

            _lastCameraPos = currentPos;
            _lastCameraRot = currentRot;

            float posIntensity = Mathf.Clamp01(posDelta / Mathf.Max(cameraMotionThreshold * 5f, 0.01f));
            float rotIntensity = Mathf.Clamp01(rotDelta / 6f);

            return Mathf.Max(posIntensity, rotIntensity);
        }

        private static bool IsCameraInsideVolume(Vector3 worldOrigin, float worldExtent, float margin)
        {
            var cam = Camera.main;
            if (cam == null) return false;
            Vector3 local = cam.transform.position - worldOrigin;
            return local.x >= -margin && local.x <= worldExtent + margin &&
                   local.y >= -margin && local.y <= worldExtent + margin &&
                   local.z >= -margin && local.z <= worldExtent + margin;
        }

        private static float GetDistanceToVolumeBounds(Vector3 worldPos, Vector3 worldOrigin, float worldExtent)
        {
            Vector3 min = worldOrigin;
            Vector3 max = worldOrigin + Vector3.one * worldExtent;

            float dx = Mathf.Max(Mathf.Max(min.x - worldPos.x, 0f), worldPos.x - max.x);
            float dy = Mathf.Max(Mathf.Max(min.y - worldPos.y, 0f), worldPos.y - max.y);
            float dz = Mathf.Max(Mathf.Max(min.z - worldPos.z, 0f), worldPos.z - max.z);

            return Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
}
