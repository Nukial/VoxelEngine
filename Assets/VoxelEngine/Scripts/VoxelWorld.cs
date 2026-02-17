using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;

namespace VoxelEngine
{
    /// <summary>
    /// Main voxel world manager. Creates and manages GPU buffers,
    /// coordinates rendering, simulation, and collision sub-systems.
    /// </summary>
    public class VoxelWorld : MonoBehaviour
    {
        private enum FireSimulationProfile
        {
            Fast,
            Realistic,
            Custom
        }

        private struct VoxelWrite
        {
            public int index;
            public uint value;
        }

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

        // GPU Buffers
        private GraphicsBuffer _voxelBufferA;
        private GraphicsBuffer _voxelBufferB;
        private GraphicsBuffer _brickMapBuffer;
        private bool _pingPong;

        // Rendering
        private Material _rayMarchMaterial;
        private MeshFilter _meshFilter;
        private MeshRenderer _meshRenderer;

        // Compute kernel IDs
        private int _terrainKernel;
        private int _clearKernel;
        private int _simKernel;
        private int _simClearKernel;
        private int _brickMapKernel;
        private int _heatLightKernel;

        // Frame counter for simulation randomness
        private uint _frameCount;
        private uint _simStep;
        private uint[] _cpuReadbackCache;
        private float _lastCpuReadbackTime = -999f;
        private bool _cpuReadbackReady;
        private bool _cpuReadbackRequestPending;
        private Vector3 _lastCameraPos;
        private bool _cameraPosInitialized;
        private float _simulationAccumulator;
        private float _motionBlend; // 0=still, 1=moving, smoothly interpolated
        private Quaternion _lastCameraRot;
        private Light _cachedDirectionalLight;
        private float _nextDirectionalLightSearchTime;

        // Properties for external access
        public int WorldSize => worldSize;
        public int BrickSize => brickSize;
        public float VoxelScale => voxelScale;
        public int BrickMapSize => worldSize / brickSize;
        public int TotalVoxels => worldSize * worldSize * worldSize;
        public GraphicsBuffer ReadBuffer => _pingPong ? _voxelBufferB : _voxelBufferA;
        public GraphicsBuffer WriteBuffer => _pingPong ? _voxelBufferA : _voxelBufferB;
        public GraphicsBuffer BrickMapBuffer => _brickMapBuffer;
        public Vector3 WorldOrigin => transform.position;
        public float WorldExtent => worldSize * voxelScale;
        public VoxelIndirectInstanceRenderer IndirectInstanceRenderer => indirectInstanceRenderer;

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
        private static readonly int PropReadBuffer = Shader.PropertyToID("_ReadBuffer");
        private static readonly int PropWriteBuffer = Shader.PropertyToID("_WriteBuffer");
        private static readonly int PropFrameCount = Shader.PropertyToID("_FrameCount");
        private static readonly int PropSimStep = Shader.PropertyToID("_SimStep");
        private static readonly int PropSeed = Shader.PropertyToID("_Seed");
        private static readonly int PropTerrainScale = Shader.PropertyToID("_TerrainScale");
        private static readonly int PropCaveScale = Shader.PropertyToID("_CaveScale");
        private static readonly int PropCaveThreshold = Shader.PropertyToID("_CaveThreshold");
        private static readonly int PropWaterLevel = Shader.PropertyToID("_WaterLevel");
        private static readonly int PropMaxRenderDist = Shader.PropertyToID("_MaxRenderDist");
        private static readonly int PropCullMode = Shader.PropertyToID("_Cull");
        private static readonly int PropShadowStrength = Shader.PropertyToID("_ShadowStrength");
        private static readonly int PropFireSpreadNeighborTemp = Shader.PropertyToID("_FireSpreadNeighborTemp");
        private static readonly int PropWoodCharTemp = Shader.PropertyToID("_WoodCharTemp");
        private static readonly int PropLeafBurnTemp = Shader.PropertyToID("_LeafBurnTemp");
        private static readonly int PropCoalBurnoutTemp = Shader.PropertyToID("_CoalBurnoutTemp");
        private static readonly int PropSnowMeltTemp = Shader.PropertyToID("_SnowMeltTemp");
        private static readonly int PropWaterEvapTemp = Shader.PropertyToID("_WaterEvapTemp");
        private static readonly int PropBurningLightTemp = Shader.PropertyToID("_BurningLightTemp");
        private static readonly int PropHeatRiseRate = Shader.PropertyToID("_HeatRiseRate");
        private static readonly int PropCoolRate = Shader.PropertyToID("_CoolRate");
        private static readonly int PropHeatSinkExtraCool = Shader.PropertyToID("_HeatSinkExtraCool");
        private static readonly int PropCoalBurnoutChanceDiv = Shader.PropertyToID("_CoalBurnoutChanceDiv");

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

            // Run simulation
            if (enableSimulation)
            {
                bool simulated = RunScheduledSimulation();
                if (simulated)
                    UpdateBrickMap();
            }

            // Update rendering properties
            UpdateRenderProperties();

            UpdateCpuReadbackCache();

            _frameCount++;
        }

        private bool RunScheduledSimulation()
        {
            float tickRate = Mathf.Max(1f, simulationTickRate);
            float tickInterval = 1f / tickRate;
            _simulationAccumulator += Time.deltaTime;

            int maxSteps = Mathf.Max(1, simulationStepsPerFrame);
            bool simulated = false;

            while (_simulationAccumulator >= tickInterval && maxSteps > 0)
            {
                RunSimulationStep();
                _simulationAccumulator -= tickInterval;
                maxSteps--;
                simulated = true;
            }

            if (_simulationAccumulator > tickInterval * 4f)
                _simulationAccumulator = tickInterval;

            return simulated;
        }

        private void UpdateCpuReadbackCache()
        {
            if (ReadBuffer == null) return;

            if (_cpuReadbackCache == null || _cpuReadbackCache.Length != TotalVoxels)
            {
                _cpuReadbackCache = new uint[TotalVoxels];
                _cpuReadbackReady = false;
            }

            if (Time.time - _lastCpuReadbackTime < cpuReadbackInterval) return;

            if (!useAsyncCpuReadback)
            {
                ReadBuffer.GetData(_cpuReadbackCache);
                _cpuReadbackReady = true;
                _lastCpuReadbackTime = Time.time;
                return;
            }

            if (_cpuReadbackRequestPending) return;

            _cpuReadbackRequestPending = true;
            _lastCpuReadbackTime = Time.time;
            AsyncGPUReadback.Request(ReadBuffer, (request) =>
            {
                _cpuReadbackRequestPending = false;
                if (!this || request.hasError) return;

                var data = request.GetData<uint>();
                if (_cpuReadbackCache == null || _cpuReadbackCache.Length != data.Length)
                    _cpuReadbackCache = new uint[data.Length];

                data.CopyTo(_cpuReadbackCache);
                _cpuReadbackReady = true;
            });
        }

        // =====================================================================
        // Initialization
        // =====================================================================

        private void Initialize()
        {
            CreateBuffers();
            CreateRenderingComponents();
            GenerateTerrain();
            UpdateBrickMap();
            UpdateRenderProperties();
        }

        private void CreateBuffers()
        {
            int total = TotalVoxels;
            int brickTotal = BrickMapSize * BrickMapSize * BrickMapSize;

            _voxelBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total, sizeof(uint));
            _voxelBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total, sizeof(uint));
            _brickMapBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, brickTotal, sizeof(uint));

            // Clear buffers
            var clearData = new uint[total];
            _voxelBufferA.SetData(clearData);
            _voxelBufferB.SetData(clearData);

            var clearBrick = new uint[brickTotal];
            _brickMapBuffer.SetData(clearBrick);

            Debug.Log($"[VoxelWorld] Created buffers: {worldSize}³ = {total:N0} voxels ({total * 4 / 1024f / 1024f:F1} MB per buffer)");
        }

        private void CreateRenderingComponents()
        {
            if (indirectInstanceRenderer == null)
                indirectInstanceRenderer = GetComponent<VoxelIndirectInstanceRenderer>();

            if (rayMarchShader == null)
            {
                rayMarchShader = Shader.Find("VoxelEngine/RayMarch");
                if (rayMarchShader == null)
                {
                    Debug.LogError("[VoxelWorld] RayMarch shader not found!");
                    return;
                }
            }

            _rayMarchMaterial = new Material(rayMarchShader);
            _rayMarchMaterial.name = "VoxelRayMarch (Runtime)";

            // Enable shadow keyword
            if (enableShadows)
                _rayMarchMaterial.EnableKeyword("VOXEL_SHADOWS_ON");
            else
                _rayMarchMaterial.DisableKeyword("VOXEL_SHADOWS_ON");

            // Create box mesh covering the voxel volume
            _meshFilter = gameObject.GetComponent<MeshFilter>();
            if (_meshFilter == null)
                _meshFilter = gameObject.AddComponent<MeshFilter>();

            _meshRenderer = gameObject.GetComponent<MeshRenderer>();
            if (_meshRenderer == null)
                _meshRenderer = gameObject.AddComponent<MeshRenderer>();

            _meshFilter.sharedMesh = CreateBoxMesh();
            _meshRenderer.sharedMaterial = _rayMarchMaterial;
            _meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _meshRenderer.receiveShadows = false;

            // Set transform to position the box
            // Box mesh is unit cube [0,1], scale to world size
            float extent = WorldExtent;
            transform.localScale = new Vector3(extent, extent, extent);
        }

        private Mesh CreateBoxMesh()
        {
            // Unit cube with margin to prevent near-plane clipping when camera
            // is near volume edges. Extends slightly beyond [0,1] on each side.
            var mesh = new Mesh();
            mesh.name = "VoxelVolume";

            const float m = 0.03f; // margin
            float lo = -m;
            float hi = 1f + m;

            Vector3[] verts = {
                // Front face
                new Vector3(lo, lo, hi), new Vector3(hi, lo, hi), new Vector3(hi, hi, hi), new Vector3(lo, hi, hi),
                // Back face
                new Vector3(hi, lo, lo), new Vector3(lo, lo, lo), new Vector3(lo, hi, lo), new Vector3(hi, hi, lo),
                // Top face
                new Vector3(lo, hi, hi), new Vector3(hi, hi, hi), new Vector3(hi, hi, lo), new Vector3(lo, hi, lo),
                // Bottom face
                new Vector3(lo, lo, lo), new Vector3(hi, lo, lo), new Vector3(hi, lo, hi), new Vector3(lo, lo, hi),
                // Right face
                new Vector3(hi, lo, hi), new Vector3(hi, lo, lo), new Vector3(hi, hi, lo), new Vector3(hi, hi, hi),
                // Left face
                new Vector3(lo, lo, lo), new Vector3(lo, lo, hi), new Vector3(lo, hi, hi), new Vector3(lo, hi, lo),
            };

            int[] tris = {
                0,2,1, 0,3,2,       // Front
                4,6,5, 4,7,6,       // Back
                8,10,9, 8,11,10,    // Top
                12,14,13, 12,15,14, // Bottom
                16,18,17, 16,19,18, // Right
                20,22,21, 20,23,22, // Left
            };

            mesh.vertices = verts;
            mesh.triangles = tris;
            mesh.RecalculateBounds();

            return mesh;
        }

        // =====================================================================
        // Terrain Generation
        // =====================================================================

        private void GenerateTerrain()
        {
            if (terrainGenShader == null)
            {
                Debug.LogError("[VoxelWorld] TerrainGeneration compute shader not assigned!");
                return;
            }

            _terrainKernel = terrainGenShader.FindKernel("GenerateTerrain");

            terrainGenShader.SetBuffer(_terrainKernel, PropVoxelBuffer, _voxelBufferA);
            terrainGenShader.SetInt(PropWorldSize, worldSize);
            terrainGenShader.SetFloat(PropSeed, seed);
            terrainGenShader.SetFloat(PropTerrainScale, terrainScale);
            terrainGenShader.SetFloat(PropCaveScale, caveScale);
            terrainGenShader.SetFloat(PropCaveThreshold, caveThreshold);
            terrainGenShader.SetInt(PropWaterLevel, Mathf.RoundToInt(worldSize * 0.3f));

            int groups = Mathf.CeilToInt(worldSize / 4f);
            terrainGenShader.Dispatch(_terrainKernel, groups, groups, groups);

            // Copy A to B for initial state using a compute shader or manual copy
            var tempData = new uint[TotalVoxels];
            _voxelBufferA.GetData(tempData);
            _voxelBufferB.SetData(tempData);

            Debug.Log("[VoxelWorld] Terrain generated successfully");
        }

        // =====================================================================
        // Simulation
        // =====================================================================

        private void RunSimulationStep()
        {
            if (simulationShader == null) return;

            _simClearKernel = simulationShader.FindKernel("ClearWriteBuffer");
            _simKernel = simulationShader.FindKernel("SimulateVoxels");
            _heatLightKernel = simulationShader.FindKernel("PropagateHeatAndLight");

            var readBuf = ReadBuffer;
            var writeBuf = WriteBuffer;

            // Phase 1: Clear write buffer
            simulationShader.SetBuffer(_simClearKernel, PropWriteBuffer, writeBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            int clearGroups = Mathf.CeilToInt(TotalVoxels / 256f);
            simulationShader.Dispatch(_simClearKernel, clearGroups, 1, 1);

            // Phase 2: Simulate (movement + chemical reactions)
            simulationShader.SetBuffer(_simKernel, PropReadBuffer, readBuf);
            simulationShader.SetBuffer(_simKernel, PropWriteBuffer, writeBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.SetInt(PropFrameCount, (int)_frameCount);
            simulationShader.SetInt(PropSimStep, (int)_simStep);

            int simGroups = Mathf.CeilToInt(worldSize / 4f);
            simulationShader.Dispatch(_simKernel, simGroups, simGroups, simGroups);

            // Phase 3: Clear read buffer to reuse it as heat/light destination
            simulationShader.SetBuffer(_simClearKernel, PropWriteBuffer, readBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.Dispatch(_simClearKernel, clearGroups, 1, 1);

            // Phase 4: Propagate heat/light with stable read->write buffers
            // Read from simulation output (writeBuf), write final state into readBuf.
            simulationShader.SetBuffer(_heatLightKernel, PropReadBuffer, writeBuf);
            simulationShader.SetBuffer(_heatLightKernel, PropWriteBuffer, readBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.SetInt(PropFrameCount, (int)_frameCount);
            ApplyFireSimulationTuning(_heatLightKernel);
            simulationShader.Dispatch(_heatLightKernel, simGroups, simGroups, simGroups);

            // Do not swap here: final state already written back to ReadBuffer.
            _simStep++;
        }

        private void ApplyFireSimulationTuning(int kernel)
        {
            int spreadTemp = fireSpreadNeighborTemp;
            int woodTemp = woodCharTemp;
            int leafTemp = leafBurnTemp;
            int coalTemp = coalBurnoutTemp;
            int snowTemp = snowMeltTemp;
            int waterTemp = waterEvapTemp;
            int burnLightTemp = burningLightTemp;
            int riseRate = heatRiseRate;
            int decayRate = coolRate;
            int sinkCool = heatSinkExtraCool;
            int burnoutDiv = coalBurnoutChanceDiv;

            if (fireProfile == FireSimulationProfile.Fast)
            {
                spreadTemp = 7;
                woodTemp = 10;
                leafTemp = 8;
                coalTemp = 13;
                snowTemp = 3;
                waterTemp = 10;
                burnLightTemp = 8;
                riseRate = 2;
                decayRate = 1;
                sinkCool = 0;
                burnoutDiv = 5;
            }
            else if (fireProfile == FireSimulationProfile.Realistic)
            {
                spreadTemp = 10;
                woodTemp = 13;
                leafTemp = 10;
                coalTemp = 15;
                snowTemp = 5;
                waterTemp = 13;
                burnLightTemp = 10;
                riseRate = 1;
                decayRate = 1;
                sinkCool = 1;
                burnoutDiv = 12;
            }

            simulationShader.SetInt(PropFireSpreadNeighborTemp, Mathf.Clamp(spreadTemp, 4, 15));
            simulationShader.SetInt(PropWoodCharTemp, Mathf.Clamp(woodTemp, 6, 15));
            simulationShader.SetInt(PropLeafBurnTemp, Mathf.Clamp(leafTemp, 6, 15));
            simulationShader.SetInt(PropCoalBurnoutTemp, Mathf.Clamp(coalTemp, 8, 15));
            simulationShader.SetInt(PropSnowMeltTemp, Mathf.Clamp(snowTemp, 2, 8));
            simulationShader.SetInt(PropWaterEvapTemp, Mathf.Clamp(waterTemp, 8, 15));
            simulationShader.SetInt(PropBurningLightTemp, Mathf.Clamp(burnLightTemp, 6, 15));
            simulationShader.SetInt(PropHeatRiseRate, Mathf.Clamp(riseRate, 1, 3));
            simulationShader.SetInt(PropCoolRate, Mathf.Clamp(decayRate, 1, 3));
            simulationShader.SetInt(PropHeatSinkExtraCool, Mathf.Clamp(sinkCool, 0, 2));
            simulationShader.SetInt(PropCoalBurnoutChanceDiv, Mathf.Max(2, burnoutDiv));
        }

        // =====================================================================
        // Brick Map
        // =====================================================================

        private void UpdateBrickMap()
        {
            if (brickMapShader == null) return;

            _brickMapKernel = brickMapShader.FindKernel("UpdateBrickMap");

            brickMapShader.SetBuffer(_brickMapKernel, PropVoxelBuffer, ReadBuffer);
            brickMapShader.SetBuffer(_brickMapKernel, PropBrickMap, _brickMapBuffer);
            brickMapShader.SetInt(PropWorldSize, worldSize);
            brickMapShader.SetInt(PropBrickSize, brickSize);
            brickMapShader.SetInt(PropBrickMapSize, BrickMapSize);

            brickMapShader.Dispatch(_brickMapKernel, BrickMapSize, BrickMapSize, BrickMapSize);
        }

        // =====================================================================
        // Render Property Updates
        // =====================================================================

        private void UpdateRenderProperties()
        {
            if (_rayMarchMaterial == null) return;

            UpdateLightingFromUnity();

            // Smooth motion detection: interpolate between still/moving states
            float rawMotion = GetCameraMotionIntensity();
            float targetBlend = rawMotion > 0f ? Mathf.Clamp01(rawMotion) : 0f;
            _motionBlend = Mathf.Lerp(_motionBlend, targetBlend,
                Time.deltaTime * qualityTransitionSpeed);
            // Snap to 0 when very close to avoid perpetual micro-blend
            if (_motionBlend < 0.01f) _motionBlend = 0f;

            int runtimeMaxRaySteps = maxRaySteps;
            int runtimeMaxShadowSteps = maxShadowSteps;

            if (enableAdaptiveQuality && _motionBlend > 0f)
            {
                // Smoothly blend ray steps between full quality and moving quality
                runtimeMaxRaySteps = Mathf.RoundToInt(
                    Mathf.Lerp(maxRaySteps, Mathf.Min(maxRaySteps, movingRaySteps), _motionBlend));

                int movingShadowTarget = Mathf.Min(maxShadowSteps, movingShadowSteps);
                int shadowFloor = Mathf.RoundToInt(maxShadowSteps * movingShadowStepFloor);
                movingShadowTarget = Mathf.Max(movingShadowTarget, shadowFloor);
                runtimeMaxShadowSteps = Mathf.RoundToInt(
                    Mathf.Lerp(maxShadowSteps, movingShadowTarget, _motionBlend));
            }

            // Use Cull Off to avoid blind/dead angles caused by dynamic cull
            // transitions and face winding edge cases on the volume mesh.
            _rayMarchMaterial.SetFloat(PropCullMode, 0f);

            // Distance-based quality scaling
            var camera = Camera.main;
            if (camera != null)
            {
                float distToCenter = Vector3.Distance(camera.transform.position,
                    WorldOrigin + Vector3.one * WorldExtent * 0.5f);

                float effectiveMaxDist = Mathf.Max(1f, maxRenderDistance);
                float fadeStart = effectiveMaxDist * Mathf.Clamp01(renderDistanceFadeRatio);
                float fadeRange = Mathf.Max(0.001f, effectiveMaxDist - fadeStart);
                float distRatio = Mathf.Clamp01((distToCenter - fadeStart) / fadeRange);

                float qualityMult = Mathf.Lerp(1f, distanceQualityFactor, distRatio);
                runtimeMaxRaySteps = Mathf.Max(64, Mathf.RoundToInt(runtimeMaxRaySteps * qualityMult));
                runtimeMaxShadowSteps = Mathf.Max(16, Mathf.RoundToInt(runtimeMaxShadowSteps * qualityMult));
            }

            _rayMarchMaterial.SetBuffer(PropVoxelBuffer, ReadBuffer);
            _rayMarchMaterial.SetBuffer(PropBrickMap, _brickMapBuffer);
            _rayMarchMaterial.SetInt(PropWorldSize, worldSize);
            _rayMarchMaterial.SetInt(PropBrickSize, brickSize);
            _rayMarchMaterial.SetInt(PropBrickMapSize, BrickMapSize);
            _rayMarchMaterial.SetFloat(PropVoxelScale, voxelScale);
            _rayMarchMaterial.SetVector(PropWorldOrigin, WorldOrigin);
            _rayMarchMaterial.SetInt(PropMaxSteps, runtimeMaxRaySteps);
            _rayMarchMaterial.SetInt(PropMaxShadowSteps, runtimeMaxShadowSteps);
            _rayMarchMaterial.SetFloat(PropMaxRenderDist, maxRenderDistance);
            _rayMarchMaterial.SetVector(PropSunDir, sunDirection.normalized);
            _rayMarchMaterial.SetColor(PropSunColor, sunColor);
            _rayMarchMaterial.SetColor(PropAmbientColor, ambientColor);
            _rayMarchMaterial.SetFloat(PropSunIntensity, sunIntensity);
            _rayMarchMaterial.SetFloat(PropFogDensity, fogDensity);
            _rayMarchMaterial.SetColor(PropFogColor, fogColor);

            // Shadow: always keep keyword ON, use _ShadowStrength for smooth transition
            float shadowStr = 1f;
            if (enableShadows)
            {
                _rayMarchMaterial.EnableKeyword("VOXEL_SHADOWS_ON");
                if (reduceShadowsWhileMoving && !stabilizeLightingWhileMoving)
                    shadowStr = Mathf.Lerp(1f, movingShadowIntensity, _motionBlend);
            }
            else
            {
                _rayMarchMaterial.DisableKeyword("VOXEL_SHADOWS_ON");
                shadowStr = 0f;
            }
            _rayMarchMaterial.SetFloat(PropShadowStrength, shadowStr);
        }

        private void UpdateLightingFromUnity()
        {
            if (!syncWithUnityDirectionalLight)
                return;

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

            if (syncAmbientFromRenderSettings)
                ambientColor = RenderSettings.ambientLight;
        }

        /// <summary>
        /// Check if the main camera is inside the voxel volume bounding box.
        /// Used to switch cull mode for correct rendering.
        /// </summary>
        private bool IsCameraInsideVolume()
        {
            var cam = Camera.main;
            if (cam == null) return false;
            Vector3 local = cam.transform.position - WorldOrigin;
            float ext = WorldExtent;
            float margin = 0.5f; // small margin for smooth transition
            return local.x >= -margin && local.x <= ext + margin &&
                   local.y >= -margin && local.y <= ext + margin &&
                   local.z >= -margin && local.z <= ext + margin;
        }

        /// <summary>
        /// Returns a 0-1 value indicating how intensely the camera is moving.
        /// 0 = stationary, 1 = fast movement. Accounts for both position and rotation.
        /// </summary>
        private float GetCameraMotionIntensity()
        {
            var camera = Camera.main;
            if (camera == null)
                return 0f;

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

            // Normalize: position delta weighted by threshold, rotation weighted by degrees
            float posIntensity = Mathf.Clamp01(posDelta / Mathf.Max(cameraMotionThreshold * 5f, 0.01f));
            float rotIntensity = Mathf.Clamp01(rotDelta / 6f); // smoother for camera look changes

            return Mathf.Max(posIntensity, rotIntensity);
        }

        // =====================================================================
        // Voxel Modification API
        // =====================================================================

        /// <summary>
        /// Set a single voxel. This modifies the current read buffer directly.
        /// Changes take effect next frame.
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

            if (writes.Count == 0)
                return;

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

        // =====================================================================
        // CPU-side Ray Cast (for interaction)
        // =====================================================================

        public uint[] GetCpuVoxelData(bool forceRefresh = false)
        {
            if (ReadBuffer == null) return null;

            if (_cpuReadbackCache == null || _cpuReadbackCache.Length != TotalVoxels)
                _cpuReadbackCache = new uint[TotalVoxels];

            if (!Application.isPlaying)
            {
                ReadBuffer.GetData(_cpuReadbackCache);
                _cpuReadbackReady = true;
                return _cpuReadbackCache;
            }

            if (forceRefresh && !_cpuReadbackRequestPending)
            {
                ReadBuffer.GetData(_cpuReadbackCache);
                _cpuReadbackReady = true;
                _lastCpuReadbackTime = Time.time;
            }

            if (!_cpuReadbackReady)
                return null;

            return _cpuReadbackCache;
        }

        /// <summary>
        /// Perform a DDA ray cast through voxel data on CPU.
        /// Requires GPU readback - use sparingly.
        /// Returns (hitPos, normal, materialId) or null if no hit.
        /// </summary>
        public bool RaycastVoxel(Ray ray, float maxDist, out Vector3Int hitPos, out Vector3Int hitNormal, out uint hitMaterial)
        {
            hitPos = Vector3Int.zero;
            hitNormal = Vector3Int.up;
            hitMaterial = 0;

            // Convert ray to voxel space
            Vector3 origin = (ray.origin - WorldOrigin) / voxelScale;
            Vector3 dir = ray.direction;

            // AABB intersection
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

            // Use throttled CPU snapshot to avoid full GPU readback every frame.
            var readbackData = GetCpuVoxelData();
            if (readbackData == null) return false;

            // DDA in voxel space
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

                // Step
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

        // =====================================================================
        // Cleanup
        // =====================================================================

        private void Cleanup()
        {
            _voxelBufferA?.Release();
            _voxelBufferB?.Release();
            _brickMapBuffer?.Release();
            _voxelBufferA = null;
            _voxelBufferB = null;
            _brickMapBuffer = null;
            _cpuReadbackCache = null;

            if (_rayMarchMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(_rayMarchMaterial);
                else
                    DestroyImmediate(_rayMarchMaterial);
            }
        }

        // =====================================================================
        // Gizmos
        // =====================================================================

        private void OnDrawGizmosSelected()
        {
            float extent = worldSize * voxelScale;
            Vector3 center = transform.position + Vector3.one * extent * 0.5f;
            Gizmos.color = new Color(0, 1, 0, 0.3f);
            Gizmos.DrawWireCube(center, Vector3.one * extent);

            // Draw water level
            float waterY = transform.position.y + worldSize * 0.3f * voxelScale;
            Gizmos.color = new Color(0, 0.5f, 1f, 0.15f);
            Gizmos.DrawCube(
                new Vector3(center.x, waterY, center.z),
                new Vector3(extent, 0.1f, extent)
            );
        }

        // =====================================================================
        // Validation
        // =====================================================================

        private void OnValidate()
        {
            worldSize = Mathf.ClosestPowerOfTwo(Mathf.Clamp(worldSize, 32, 512));
            brickSize = Mathf.Clamp(brickSize, 4, 16);
            movingShadowStepFloor = Mathf.Clamp01(movingShadowStepFloor);

            if (worldSize % brickSize != 0)
                brickSize = 8;
        }
    }
}
