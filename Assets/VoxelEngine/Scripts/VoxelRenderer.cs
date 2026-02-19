using UnityEngine;
using UnityEngine.Rendering;

namespace VoxelEngine
{
    /// <summary>
    /// Manages the ray-march material, box mesh, and per-frame shader property updates
    /// for voxel rendering. Also handles directional-light synchronization.
    /// </summary>
    public class VoxelRenderer
    {
        // Shader property IDs (cached)
        private static readonly int PropVoxelBuffer = Shader.PropertyToID("_VoxelBuffer");
        private static readonly int PropBrickMap = Shader.PropertyToID("_BrickMap");
        private static readonly int PropWorldSize = Shader.PropertyToID("_WorldSize");
        private static readonly int PropBrickSize = Shader.PropertyToID("_BrickSize");
        private static readonly int PropBrickMapSize = Shader.PropertyToID("_BrickMapSize");
        private static readonly int PropVoxelScale = Shader.PropertyToID("_VoxelScale");
        private static readonly int PropWorldOrigin = Shader.PropertyToID("_WorldOrigin");
        private static readonly int PropMaxSteps = Shader.PropertyToID("_MaxSteps");
        private static readonly int PropMaxShadowSteps = Shader.PropertyToID("_MaxShadowSteps");
        private static readonly int PropSunDir = Shader.PropertyToID("_SunDir");
        private static readonly int PropSunColor = Shader.PropertyToID("_SunColor");
        private static readonly int PropAmbientColor = Shader.PropertyToID("_AmbientColor");
        private static readonly int PropSunIntensity = Shader.PropertyToID("_SunIntensity");
        private static readonly int PropFogDensity = Shader.PropertyToID("_FogDensity");
        private static readonly int PropFogColor = Shader.PropertyToID("_FogColor");
        private static readonly int PropMaxRenderDist = Shader.PropertyToID("_MaxRenderDist");
        private static readonly int PropShadowStrength = Shader.PropertyToID("_ShadowStrength");
        private static readonly int PropFastLightingMaxDist = Shader.PropertyToID("_FastLightingMaxDist");
        private static readonly int PropShadowRayMaxDist = Shader.PropertyToID("_ShadowRayMaxDist");

        private Material _rayMarchMaterial;
        private Mesh _boxMesh;
        private MeshFilter _meshFilter;
        private MeshRenderer _meshRenderer;
        private int _layer;

        // Lighting state
        private Light _cachedDirectionalLight;
        private float _nextDirectionalLightSearchTime;

        // Smoothed shadow params
        private float _smoothedShadowRayMaxDist;
        private float _smoothedFastLightingMaxDist;
        private bool _smoothedShadowParamsInit;

        public Material Material => _rayMarchMaterial;

        /// <summary>
        /// Create the ray-march material, box mesh, and attach rendering components.
        /// </summary>
        public void Initialize(GameObject go, Shader rayMarchShader, bool enableShadows,
            float worldExtent, ref VoxelIndirectInstanceRenderer indirectInstanceRenderer)
        {
            _smoothedShadowParamsInit = false;

            if (indirectInstanceRenderer == null)
                indirectInstanceRenderer = go.GetComponent<VoxelIndirectInstanceRenderer>();

            Shader shader = rayMarchShader;
            if (shader == null)
            {
                shader = Shader.Find("VoxelEngine/RayMarch");
                if (shader == null)
                {
                    Debug.LogError("[VoxelRenderer] RayMarch shader not found!");
                    return;
                }
            }

            _rayMarchMaterial = new Material(shader);
            _rayMarchMaterial.name = "VoxelRayMarch (Runtime)";

            if (enableShadows)
                _rayMarchMaterial.EnableKeyword("VOXEL_SHADOWS_ON");
            else
                _rayMarchMaterial.DisableKeyword("VOXEL_SHADOWS_ON");

            _boxMesh = CreateBoxMesh();
            _layer = go.layer;

            // Use MeshFilter + MeshRenderer as the primary rendering path
            _meshFilter = go.GetComponent<MeshFilter>();
            if (_meshFilter == null)
                _meshFilter = go.AddComponent<MeshFilter>();

            _meshRenderer = go.GetComponent<MeshRenderer>();
            if (_meshRenderer == null)
                _meshRenderer = go.AddComponent<MeshRenderer>();

            _meshFilter.sharedMesh = _boxMesh;
            _meshRenderer.sharedMaterial = _rayMarchMaterial;
            _meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _meshRenderer.receiveShadows = false;
            _meshRenderer.allowOcclusionWhenDynamic = false;

            go.transform.localScale = new Vector3(worldExtent, worldExtent, worldExtent);
        }

        /// <summary>
        /// Update all shader properties each frame.
        /// </summary>
        public void UpdateProperties(VoxelBufferManager buffers, VoxelAdaptiveQuality quality,
            Vector3 worldOrigin, float voxelScale, float maxRenderDistance,
            bool enableShadows, bool reduceShadowsWhileMoving, bool stabilizeLightingWhileMoving,
            float movingShadowIntensity, float fastLightingDistanceRatio, float shadowRayDistanceRatio,
            float qualityTransitionSpeed,
            Color sunColor, Vector3 sunDirection, float sunIntensity,
            Color ambientColor, float fogDensity, Color fogColor)
        {
            if (_rayMarchMaterial == null) return;

            int runtimeMaxRaySteps = quality.RuntimeMaxRaySteps;
            int runtimeMaxShadowSteps = quality.RuntimeMaxShadowSteps;
            float motionBlend = quality.MotionBlend;
            float gpuLoadBlend = quality.GpuLoadBlend;

            _rayMarchMaterial.SetBuffer(PropVoxelBuffer, buffers.ReadBuffer);
            _rayMarchMaterial.SetBuffer(PropBrickMap, buffers.BrickMapBuffer);
            _rayMarchMaterial.SetInt(PropWorldSize, buffers.WorldSize);
            _rayMarchMaterial.SetInt(PropBrickSize, buffers.BrickSize);
            _rayMarchMaterial.SetInt(PropBrickMapSize, buffers.BrickMapSize);
            _rayMarchMaterial.SetFloat(PropVoxelScale, voxelScale);
            _rayMarchMaterial.SetVector(PropWorldOrigin, worldOrigin);
            _rayMarchMaterial.SetInt(PropMaxSteps, runtimeMaxRaySteps);
            _rayMarchMaterial.SetInt(PropMaxShadowSteps, runtimeMaxShadowSteps);
            _rayMarchMaterial.SetFloat(PropMaxRenderDist, maxRenderDistance);

            // Lighting distance parameters with temporal smoothing
            float fastLightingDist = Mathf.Max(6f, maxRenderDistance * fastLightingDistanceRatio);
            float shadowRayMaxDist = Mathf.Max(4f, maxRenderDistance * shadowRayDistanceRatio);
            if (quality.EnableGpuAdaptive && gpuLoadBlend > 0f)
            {
                fastLightingDist *= Mathf.Lerp(1f, 0.55f, gpuLoadBlend);
                shadowRayMaxDist *= Mathf.Lerp(1f, 0.5f, gpuLoadBlend);
            }

            if (!_smoothedShadowParamsInit)
            {
                _smoothedFastLightingMaxDist = fastLightingDist;
                _smoothedShadowRayMaxDist = shadowRayMaxDist;
                _smoothedShadowParamsInit = true;
            }
            else
            {
                float shadowSmoothSpeed = Time.deltaTime * qualityTransitionSpeed * 0.35f;
                _smoothedFastLightingMaxDist = Mathf.Lerp(_smoothedFastLightingMaxDist, fastLightingDist, shadowSmoothSpeed);
                _smoothedShadowRayMaxDist = Mathf.Lerp(_smoothedShadowRayMaxDist, shadowRayMaxDist, shadowSmoothSpeed);
            }

            _rayMarchMaterial.SetFloat(PropFastLightingMaxDist, _smoothedFastLightingMaxDist);
            _rayMarchMaterial.SetFloat(PropShadowRayMaxDist, _smoothedShadowRayMaxDist);
            _rayMarchMaterial.SetVector(PropSunDir, sunDirection.normalized);
            _rayMarchMaterial.SetColor(PropSunColor, sunColor);
            _rayMarchMaterial.SetColor(PropAmbientColor, ambientColor);
            _rayMarchMaterial.SetFloat(PropSunIntensity, sunIntensity);
            _rayMarchMaterial.SetFloat(PropFogDensity, fogDensity);
            _rayMarchMaterial.SetColor(PropFogColor, fogColor);

            // Shadow strength
            float shadowStr = 1f;
            if (enableShadows)
            {
                _rayMarchMaterial.EnableKeyword("VOXEL_SHADOWS_ON");
                if (reduceShadowsWhileMoving && !stabilizeLightingWhileMoving)
                    shadowStr = Mathf.Lerp(1f, movingShadowIntensity, motionBlend);
            }
            else
            {
                _rayMarchMaterial.DisableKeyword("VOXEL_SHADOWS_ON");
                shadowStr = 0f;
            }
            _rayMarchMaterial.SetFloat(PropShadowStrength, shadowStr);
        }

        /// <summary>
        /// Sync sun direction, color, and intensity from Unity's directional light.
        /// </summary>
        public void SyncLightingFromUnity(bool syncEnabled, Light directionalLightOverride,
            bool syncAmbient, ref Color ambientColor, ref Color sunColor,
            ref Vector3 sunDirection, ref float sunIntensity)
        {
            if (!syncEnabled) return;

            Light source = directionalLightOverride;
            if (source == null)
            {
                if (_cachedDirectionalLight == null)
                {
                    if (Time.time >= _nextDirectionalLightSearchTime)
                    {
                        _cachedDirectionalLight = RenderSettings.sun;
                        _nextDirectionalLightSearchTime = Time.time + 1f;
                    }
                }
                else
                {
                    if (!Application.isPlaying || !_cachedDirectionalLight.isActiveAndEnabled)
                        _cachedDirectionalLight = null;
                }

                source = _cachedDirectionalLight;
            }

            if (source == null || source.type != LightType.Directional || !source.isActiveAndEnabled)
                return;

            sunDirection = -source.transform.forward;
            sunColor = source.color;
            sunIntensity = source.intensity;

            if (syncAmbient)
                ambientColor = RenderSettings.ambientLight;
        }

        /// <summary>
        /// Backup draw call via Graphics.DrawMesh — completely bypasses Unity's
        /// frustum culling. Called every frame from LateUpdate as a safety net
        /// in case the MeshRenderer path is culled.
        /// </summary>
        public void DrawManual(Matrix4x4 localToWorld)
        {
            if (_boxMesh == null || _rayMarchMaterial == null) return;

            // If MeshRenderer is still active, disable it so we don't double-render.
            // Graphics.DrawMesh is the more reliable path.
            if (_meshRenderer != null && _meshRenderer.enabled)
                _meshRenderer.enabled = false;

            Graphics.DrawMesh(
                _boxMesh,
                localToWorld,
                _rayMarchMaterial,
                _layer,
                null,   // camera — null = all cameras
                0,      // submesh index
                null,   // MaterialPropertyBlock
                ShadowCastingMode.Off,
                false,  // receive shadows
                null,   // probe anchor
                LightProbeUsage.Off
            );
        }

        public void Cleanup()
        {
            _smoothedShadowParamsInit = false;
            if (_rayMarchMaterial != null)
            {
                if (Application.isPlaying)
                    Object.Destroy(_rayMarchMaterial);
                else
                    Object.DestroyImmediate(_rayMarchMaterial);
            }
            _rayMarchMaterial = null;
            _boxMesh = null;
            _meshFilter = null;
            _meshRenderer = null;
        }

        // ----- helpers -----

        private Mesh CreateBoxMesh()
        {
            var mesh = new Mesh();
            mesh.name = "VoxelVolume";

            const float m = 0.12f;
            float lo = -m;
            float hi = 1f + m;

            Vector3[] verts = {
                new Vector3(lo, lo, hi), new Vector3(hi, lo, hi), new Vector3(hi, hi, hi), new Vector3(lo, hi, hi),
                new Vector3(hi, lo, lo), new Vector3(lo, lo, lo), new Vector3(lo, hi, lo), new Vector3(hi, hi, lo),
                new Vector3(lo, hi, hi), new Vector3(hi, hi, hi), new Vector3(hi, hi, lo), new Vector3(lo, hi, lo),
                new Vector3(lo, lo, lo), new Vector3(hi, lo, lo), new Vector3(hi, lo, hi), new Vector3(lo, lo, hi),
                new Vector3(hi, lo, hi), new Vector3(hi, lo, lo), new Vector3(hi, hi, lo), new Vector3(hi, hi, hi),
                new Vector3(lo, lo, lo), new Vector3(lo, lo, hi), new Vector3(lo, hi, hi), new Vector3(lo, hi, lo),
            };

            int[] tris = {
                0,2,1, 0,3,2,
                4,6,5, 4,7,6,
                8,10,9, 8,11,10,
                12,14,13, 12,15,14,
                16,18,17, 16,19,18,
                20,22,21, 20,23,22,
            };

            mesh.vertices = verts;
            mesh.triangles = tris;
            // Use massive bounds so Graphics.DrawMesh never frustum-culls
            // this mesh — especially when the camera is inside the volume.
            mesh.bounds = new Bounds(new Vector3(0.5f, 0.5f, 0.5f), Vector3.one * 100000f);
            return mesh;
        }
    }
}
