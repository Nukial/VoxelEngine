using UnityEngine;
using System.Collections.Generic;

namespace VoxelEngine
{
    /// <summary>
    /// Main voxel world coordinator. Owns serialized configuration and delegates
    /// to focused sub-systems: buffers, terrain, simulation, rendering, quality, and readback.
    /// The public API (SetVoxel, RaycastVoxel, etc.) is preserved for external scripts.
    /// </summary>
    public class VoxelWorld : MonoBehaviour
    {
        private struct VoxelWrite
        {
            public int index;
            public uint value;
        }

        // =================================================================
        // Serialized Configuration
        // =================================================================

        [Header("World Configuration")]
        [SerializeField] private int worldSize = 128;
        [SerializeField] private int brickSize = 8;
        [SerializeField] private float voxelScale = 0.25f;

        [Header("Terrain Generation")]
        [SerializeField] private ComputeShader terrainGenShader;
        [SerializeField] private float terrainScale = 0.02f;
        [SerializeField] private float caveScale = 0.06f;
        [SerializeField] [Range(0.5f, 0.9f)] private float caveThreshold = 0.72f;
        [SerializeField] private float seed = 42f;

        [Header("Simulation")]
        [SerializeField] private ComputeShader simulationShader;
        [SerializeField] private ComputeShader brickMapShader;
        [SerializeField] private bool enableSimulation = true;
        [SerializeField] [Range(1, 8)] private int simulationStepsPerFrame = 1;
        [SerializeField] [Range(5f, 120f)] private float simulationTickRate = 30f;
        [SerializeField] [Range(1, 8)] private int lightPropagationPasses = 5;

        [Header("Fire Simulation")]
        [SerializeField] private FireSimulationProfile fireProfile = FireSimulationProfile.Realistic;
        [SerializeField] [Range(4, 14)] private int fireSpreadNeighborTemp = 10;
        [SerializeField] [Range(6, 15)] private int woodCharTemp = 13;
        [SerializeField] [Range(6, 15)] private int leafBurnTemp = 10;
        [SerializeField] [Range(8, 15)] private int coalBurnoutTemp = 15;
        [SerializeField] [Range(2, 8)] private int snowMeltTemp = 5;
        [SerializeField] [Range(8, 15)] private int waterEvapTemp = 13;
        [SerializeField] [Range(6, 15)] private int burningLightTemp = 10;
        [SerializeField] [Range(1, 3)] private int heatRiseRate = 1;
        [SerializeField] [Range(1, 3)] private int coolRate = 1;
        [SerializeField] [Range(0, 2)] private int heatSinkExtraCool = 1;
        [SerializeField] [Range(2, 20)] private int coalBurnoutChanceDiv = 12;

        [Header("Rendering")]
        [SerializeField] private Shader rayMarchShader;
        [SerializeField] private VoxelIndirectInstanceRenderer indirectInstanceRenderer;
        [SerializeField] private int maxRaySteps = 512;
        [SerializeField] private int maxShadowSteps = 128;
        [SerializeField] private bool enableShadows = true;

        [Header("View Distance")]
        [SerializeField] private float maxRenderDistance = 80f;
        [SerializeField] [Range(0.3f, 1.0f)] private float distanceQualityFactor = 0.6f;
        [SerializeField] private float renderDistanceFadeRatio = 0.8f;

        [Header("Edge Performance")]
        [SerializeField] private bool enableEdgeLoadGuard = true;
        [SerializeField] [Min(0.1f)] private float edgeLoadDistance = 8f;
        [SerializeField] [Range(0.35f, 1.0f)] private float edgeRayStepScale = 0.7f;
        [SerializeField] [Range(0.2f, 1.0f)] private float edgeShadowStepScale = 0.65f;

        [Header("Adaptive Quality")]
        [SerializeField] private bool enableAdaptiveQuality = true;
        [SerializeField] [Range(64, 512)] private int movingRaySteps = 192;
        [SerializeField] [Range(16, 256)] private int movingShadowSteps = 48;
        [SerializeField] [Range(0.4f, 1.0f)] private float movingShadowStepFloor = 0.75f;
        [SerializeField] [Min(0.001f)] private float cameraMotionThreshold = 0.02f;
        [SerializeField] private bool reduceShadowsWhileMoving = true;
        [SerializeField] private bool stabilizeLightingWhileMoving = true;
        [SerializeField] [Range(0.1f, 0.5f)] private float movingShadowIntensity = 0.35f;
        [SerializeField] [Range(2f, 15f)] private float qualityTransitionSpeed = 6f;

        [Header("GPU Load Adaptive")]
        [SerializeField] private bool enableGpuAdaptiveQuality = true;
        [SerializeField] [Range(30f, 165f)] private float gpuTargetFrameRate = 60f;
        [SerializeField] [Range(0.35f, 0.95f)] private float gpuRayStepFloor = 0.58f;
        [SerializeField] [Range(0.2f, 0.9f)] private float gpuShadowStepFloor = 0.45f;
        [SerializeField] [Range(0.2f, 0.9f)] private float fastLightingDistanceRatio = 0.52f;
        [SerializeField] [Range(0.15f, 0.8f)] private float shadowRayDistanceRatio = 0.45f;

        [Header("Lighting")]
        [SerializeField] private bool syncWithUnityDirectionalLight = true;
        [SerializeField] private Light directionalLightOverride;
        [SerializeField] private bool syncAmbientFromRenderSettings = false;
        [SerializeField] private Color ambientColor = new Color(0.15f, 0.18f, 0.25f);
        [SerializeField] private Color sunColor = new Color(1f, 0.95f, 0.85f);
        [SerializeField] private float sunIntensity = 1.2f;
        [SerializeField] private Vector3 sunDirection = new Vector3(0.5f, 0.8f, 0.3f);

        [Header("Fog")]
        [SerializeField] private float fogDensity = 0.003f;
        [SerializeField] private Color fogColor = new Color(0.6f, 0.75f, 0.9f);

        [Header("CPU Readback")]
        [SerializeField] [Min(0.01f)] private float cpuReadbackInterval = 0.08f;
        [SerializeField] private bool useAsyncCpuReadback = true;

        // =================================================================
        // Sub-systems (plain C# objects — no extra MonoBehaviours needed)
        // =================================================================

        private VoxelBufferManager _buffers;
        private VoxelTerrainGenerator _terrain;
        private VoxelSimulation _simulation;
        private VoxelRenderer _renderer;
        private VoxelAdaptiveQuality _quality;
        private VoxelCpuReadback _cpuReadback;

        private uint _frameCount;

        // =================================================================
        // Public API (unchanged — external scripts keep working)
        // =================================================================

        public int WorldSize => worldSize;
        public int BrickSize => brickSize;
        public float VoxelScale => voxelScale;
        public int BrickMapSize => worldSize / brickSize;
        public int TotalVoxels => worldSize * worldSize * worldSize;
        public GraphicsBuffer ReadBuffer => _buffers?.ReadBuffer;
        public GraphicsBuffer WriteBuffer => _buffers?.WriteBuffer;
        public GraphicsBuffer BrickMapBuffer => _buffers?.BrickMapBuffer;
        public Vector3 WorldOrigin => transform.position;
        public float WorldExtent => worldSize * voxelScale;
        public VoxelIndirectInstanceRenderer IndirectInstanceRenderer => indirectInstanceRenderer;

        // =================================================================
        // Lifecycle
        // =================================================================

        private void OnEnable()
        {
            Initialize();
        }

        private void OnDisable()
        {
            Cleanup();
        }

        private void Update()
        {
            if (!Application.isPlaying) return;

            if (enableSimulation)
            {
                bool simulated = _simulation.RunScheduled(
                    simulationShader, brickMapShader, _buffers, _frameCount,
                    simulationTickRate, simulationStepsPerFrame, lightPropagationPasses,
                    BuildFireSettings());

                if (simulated)
                    _simulation.UpdateBrickMap(brickMapShader, _buffers);
            }

            // Adaptive quality
            _quality.Update(
                maxRaySteps, maxShadowSteps,
                enableAdaptiveQuality, movingRaySteps, movingShadowSteps,
                movingShadowStepFloor, cameraMotionThreshold, qualityTransitionSpeed,
                enableGpuAdaptiveQuality, gpuTargetFrameRate,
                gpuRayStepFloor, gpuShadowStepFloor,
                enableEdgeLoadGuard, edgeLoadDistance,
                edgeRayStepScale, edgeShadowStepScale,
                maxRenderDistance, distanceQualityFactor, renderDistanceFadeRatio,
                WorldOrigin, WorldExtent);

            // Lighting sync
            _renderer.SyncLightingFromUnity(syncWithUnityDirectionalLight, directionalLightOverride,
                syncAmbientFromRenderSettings, ref ambientColor, ref sunColor,
                ref sunDirection, ref sunIntensity);

            // Render properties
            _renderer.UpdateProperties(_buffers, _quality,
                WorldOrigin, voxelScale, maxRenderDistance,
                enableShadows, reduceShadowsWhileMoving, stabilizeLightingWhileMoving,
                movingShadowIntensity, fastLightingDistanceRatio, shadowRayDistanceRatio,
                qualityTransitionSpeed,
                sunColor, sunDirection, sunIntensity,
                ambientColor, fogDensity, fogColor);

            // CPU readback
            _cpuReadback.UpdateCache(ReadBuffer, TotalVoxels, cpuReadbackInterval, useAsyncCpuReadback);

            _frameCount++;
        }

        private void LateUpdate()
        {
            if (!Application.isPlaying) return;

            // Draw the voxel volume manually — bypasses Unity's frustum culling
            // so the map is never culled when the camera enters the bounding box.
            _renderer?.DrawManual(transform.localToWorldMatrix);
        }

        // =================================================================
        // Initialization / Cleanup
        // =================================================================

        private void Initialize()
        {
            _buffers = new VoxelBufferManager();
            _terrain = new VoxelTerrainGenerator();
            _simulation = new VoxelSimulation();
            _renderer = new VoxelRenderer();
            _quality = new VoxelAdaptiveQuality();
            _cpuReadback = new VoxelCpuReadback();

            _buffers.Create(worldSize, brickSize);
            _simulation.CacheKernelIDs(simulationShader, brickMapShader);
            _renderer.Initialize(gameObject, rayMarchShader, enableShadows, WorldExtent,
                ref indirectInstanceRenderer);
            _terrain.Generate(terrainGenShader, _buffers, seed, terrainScale, caveScale, caveThreshold);
            _simulation.UpdateBrickMap(brickMapShader, _buffers);

            // Initial render update
            _quality.Update(
                maxRaySteps, maxShadowSteps,
                enableAdaptiveQuality, movingRaySteps, movingShadowSteps,
                movingShadowStepFloor, cameraMotionThreshold, qualityTransitionSpeed,
                enableGpuAdaptiveQuality, gpuTargetFrameRate,
                gpuRayStepFloor, gpuShadowStepFloor,
                enableEdgeLoadGuard, edgeLoadDistance,
                edgeRayStepScale, edgeShadowStepScale,
                maxRenderDistance, distanceQualityFactor, renderDistanceFadeRatio,
                WorldOrigin, WorldExtent);

            _renderer.UpdateProperties(_buffers, _quality,
                WorldOrigin, voxelScale, maxRenderDistance,
                enableShadows, reduceShadowsWhileMoving, stabilizeLightingWhileMoving,
                movingShadowIntensity, fastLightingDistanceRatio, shadowRayDistanceRatio,
                qualityTransitionSpeed,
                sunColor, sunDirection, sunIntensity,
                ambientColor, fogDensity, fogColor);
        }

        private void Cleanup()
        {
            _renderer?.Cleanup();
            _buffers?.Release();
            _cpuReadback?.Reset();
            _simulation?.Reset();
        }

        // =================================================================
        // Voxel Modification API
        // =================================================================

        /// <summary>
        /// Set a single voxel. Changes take effect next frame.
        /// </summary>
        public void SetVoxel(Vector3Int pos, uint voxelData)
        {
            if (!VoxelData.IsInBounds(pos, worldSize)) return;
            int idx = VoxelData.Flatten3D(pos, worldSize);
            ReadBuffer.SetData(new uint[] { voxelData }, 0, idx, 1);
        }

        /// <summary>
        /// Set a sphere of voxels to a given material.
        /// </summary>
        public void SetVoxelSphere(Vector3 center, float radius, uint materialId)
        {
            int r = Mathf.CeilToInt(radius);
            var writes = new List<VoxelWrite>();
            uint voxelValue = materialId == VoxelData.MAT_AIR ? 0u : VoxelData.PackWithDefaultColor(materialId);

            for (int z = -r; z <= r; z++)
            for (int y = -r; y <= r; y++)
            for (int x = -r; x <= r; x++)
            {
                Vector3Int pos = new Vector3Int(
                    Mathf.RoundToInt(center.x) + x,
                    Mathf.RoundToInt(center.y) + y,
                    Mathf.RoundToInt(center.z) + z
                );

                if (!VoxelData.IsInBounds(pos, worldSize)) continue;
                if (new Vector3(x, y, z).magnitude > radius) continue;

                int idx = VoxelData.Flatten3D(pos, worldSize);
                writes.Add(new VoxelWrite { index = idx, value = voxelValue });
            }

            if (writes.Count == 0) return;

            writes.Sort((a, b) => a.index.CompareTo(b.index));

            int count = writes.Count;
            var indices = new int[count];
            var values = new uint[count];
            for (int i = 0; i < count; i++)
            {
                indices[i] = writes[i].index;
                values[i] = writes[i].value;
            }

            int runStart = 0;
            while (runStart < count)
            {
                int runEnd = runStart + 1;
                while (runEnd < count && indices[runEnd] == indices[runEnd - 1] + 1)
                    runEnd++;

                int gpuStart = indices[runStart];
                int runLength = runEnd - runStart;
                ReadBuffer.SetData(values, runStart, gpuStart, runLength);
                runStart = runEnd;
            }
        }

        /// <summary>
        /// Convert world position to voxel coordinate.
        /// </summary>
        public Vector3Int WorldToVoxel(Vector3 worldPos)
        {
            Vector3 local = (worldPos - WorldOrigin) / voxelScale;
            return new Vector3Int(
                Mathf.FloorToInt(local.x),
                Mathf.FloorToInt(local.y),
                Mathf.FloorToInt(local.z)
            );
        }

        /// <summary>
        /// Convert voxel coordinate to world position (center of voxel).
        /// </summary>
        public Vector3 VoxelToWorldPos(Vector3Int voxelPos)
        {
            return new Vector3(voxelPos.x + 0.5f, voxelPos.y + 0.5f, voxelPos.z + 0.5f) * voxelScale + WorldOrigin;
        }

        // =================================================================
        // CPU-side Ray Cast (for interaction)
        // =================================================================

        public uint[] GetCpuVoxelData(bool forceRefresh = false)
        {
            return _cpuReadback.GetData(ReadBuffer, TotalVoxels,
                cpuReadbackInterval, useAsyncCpuReadback, forceRefresh);
        }

        /// <summary>
        /// DDA ray cast through voxel data on CPU.
        /// Returns true on hit with hitPos, hitNormal, hitMaterial populated.
        /// </summary>
        public bool RaycastVoxel(Ray ray, float maxDist,
            out Vector3Int hitPos, out Vector3Int hitNormal, out uint hitMaterial)
        {
            hitPos = Vector3Int.zero;
            hitNormal = Vector3Int.up;
            hitMaterial = 0;

            Vector3 origin = (ray.origin - WorldOrigin) / voxelScale;
            Vector3 dir = ray.direction;

            Vector3 invDir = new Vector3(
                Mathf.Abs(dir.x) < 1e-8f ? 1e8f * Mathf.Sign(dir.x + 1e-10f) : 1f / dir.x,
                Mathf.Abs(dir.y) < 1e-8f ? 1e8f * Mathf.Sign(dir.y + 1e-10f) : 1f / dir.y,
                Mathf.Abs(dir.z) < 1e-8f ? 1e8f * Mathf.Sign(dir.z + 1e-10f) : 1f / dir.z
            );

            Vector3 t0 = Vector3.Scale(-origin, invDir);
            Vector3 t1 = Vector3.Scale(new Vector3(worldSize, worldSize, worldSize) - origin, invDir);

            Vector3 tmin = Vector3.Min(t0, t1);
            Vector3 tmax = Vector3.Max(t0, t1);

            float tNear = Mathf.Max(Mathf.Max(tmin.x, tmin.y), tmin.z);
            float tFar = Mathf.Min(Mathf.Min(tmax.x, tmax.y), tmax.z);

            if (tNear > tFar || tFar < 0) return false;
            tNear = Mathf.Max(tNear, 0.001f);

            var readbackData = GetCpuVoxelData();
            if (readbackData == null) return false;

            Vector3 startPos = origin + dir * tNear;
            startPos = Vector3.Max(startPos, Vector3.one * 0.001f);
            startPos = Vector3.Min(startPos, Vector3.one * (worldSize - 0.001f));

            Vector3Int pos = new Vector3Int(
                Mathf.FloorToInt(startPos.x),
                Mathf.FloorToInt(startPos.y),
                Mathf.FloorToInt(startPos.z)
            );
            pos = Vector3Int.Max(pos, Vector3Int.zero);
            pos = Vector3Int.Min(pos, new Vector3Int(worldSize - 1, worldSize - 1, worldSize - 1));

            Vector3Int step = new Vector3Int(
                dir.x >= 0 ? 1 : -1,
                dir.y >= 0 ? 1 : -1,
                dir.z >= 0 ? 1 : -1
            );

            Vector3 tDelta = new Vector3(
                Mathf.Abs(invDir.x),
                Mathf.Abs(invDir.y),
                Mathf.Abs(invDir.z)
            );

            Vector3 tMaxV = new Vector3(
                (step.x > 0 ? pos.x + 1 - origin.x : origin.x - pos.x) * Mathf.Abs(invDir.x),
                (step.y > 0 ? pos.y + 1 - origin.y : origin.y - pos.y) * Mathf.Abs(invDir.y),
                (step.z > 0 ? pos.z + 1 - origin.z : origin.z - pos.z) * Mathf.Abs(invDir.z)
            );

            Vector3Int normal = Vector3Int.zero;
            int maxSteps = Mathf.Min(512, Mathf.RoundToInt(maxDist / voxelScale));

            for (int i = 0; i < maxSteps; i++)
            {
                if (pos.x < 0 || pos.x >= worldSize ||
                    pos.y < 0 || pos.y >= worldSize ||
                    pos.z < 0 || pos.z >= worldSize)
                    break;

                int idx = VoxelData.Flatten3D(pos, worldSize);
                uint voxel = readbackData[idx];
                uint mat = VoxelData.GetMaterialId(voxel);

                if (mat != VoxelData.MAT_AIR)
                {
                    hitPos = pos;
                    hitNormal = normal;
                    hitMaterial = mat;
                    return true;
                }

                if (tMaxV.x < tMaxV.y)
                {
                    if (tMaxV.x < tMaxV.z)
                    {
                        pos.x += step.x;
                        tMaxV.x += tDelta.x;
                        normal = new Vector3Int(-step.x, 0, 0);
                    }
                    else
                    {
                        pos.z += step.z;
                        tMaxV.z += tDelta.z;
                        normal = new Vector3Int(0, 0, -step.z);
                    }
                }
                else
                {
                    if (tMaxV.y < tMaxV.z)
                    {
                        pos.y += step.y;
                        tMaxV.y += tDelta.y;
                        normal = new Vector3Int(0, -step.y, 0);
                    }
                    else
                    {
                        pos.z += step.z;
                        tMaxV.z += tDelta.z;
                        normal = new Vector3Int(0, 0, -step.z);
                    }
                }
            }

            return false;
        }

        // =================================================================
        // Gizmos
        // =================================================================

        private void OnDrawGizmosSelected()
        {
            float extent = worldSize * voxelScale;
            Vector3 center = transform.position + Vector3.one * extent * 0.5f;
            Gizmos.color = new Color(0, 1, 0, 0.3f);
            Gizmos.DrawWireCube(center, Vector3.one * extent);

            float waterY = transform.position.y + worldSize * 0.3f * voxelScale;
            Gizmos.color = new Color(0, 0.5f, 1f, 0.15f);
            Gizmos.DrawCube(
                new Vector3(center.x, waterY, center.z),
                new Vector3(extent, 0.1f, extent)
            );
        }

        // =================================================================
        // Validation
        // =================================================================

        private void OnValidate()
        {
            worldSize = Mathf.ClosestPowerOfTwo(Mathf.Clamp(worldSize, 32, 512));
            brickSize = Mathf.Clamp(brickSize, 4, 16);
            movingShadowStepFloor = Mathf.Clamp01(movingShadowStepFloor);
            gpuRayStepFloor = Mathf.Clamp(gpuRayStepFloor, 0.35f, 0.95f);
            gpuShadowStepFloor = Mathf.Clamp(gpuShadowStepFloor, 0.2f, 0.9f);
            fastLightingDistanceRatio = Mathf.Clamp(fastLightingDistanceRatio, 0.2f, 0.9f);
            shadowRayDistanceRatio = Mathf.Clamp(shadowRayDistanceRatio, 0.15f, 0.8f);
            gpuTargetFrameRate = Mathf.Clamp(gpuTargetFrameRate, 30f, 165f);
            edgeLoadDistance = Mathf.Max(0.1f, edgeLoadDistance);
            edgeRayStepScale = Mathf.Clamp(edgeRayStepScale, 0.35f, 1f);
            edgeShadowStepScale = Mathf.Clamp(edgeShadowStepScale, 0.2f, 1f);

            if (worldSize % brickSize != 0)
                brickSize = 8;
        }

        // =================================================================
        // Helpers
        // =================================================================

        private FireSimulationSettings BuildFireSettings()
        {
            return new FireSimulationSettings
            {
                profile = fireProfile,
                fireSpreadNeighborTemp = fireSpreadNeighborTemp,
                woodCharTemp = woodCharTemp,
                leafBurnTemp = leafBurnTemp,
                coalBurnoutTemp = coalBurnoutTemp,
                snowMeltTemp = snowMeltTemp,
                waterEvapTemp = waterEvapTemp,
                burningLightTemp = burningLightTemp,
                heatRiseRate = heatRiseRate,
                coolRate = coolRate,
                heatSinkExtraCool = heatSinkExtraCool,
                coalBurnoutChanceDiv = coalBurnoutChanceDiv
            };
        }
    }
}
