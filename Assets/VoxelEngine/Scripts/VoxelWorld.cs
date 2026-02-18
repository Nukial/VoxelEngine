using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

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
        [SerializeField] private ComputeShader svoBuildShader;
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

        [Header("CPU Simulation (Jobs + Burst)")]
        [Tooltip("Run simulation on CPU via Burst jobs instead of GPU compute shaders. " +
                 "Frees the GPU for rendering when GPU is the bottleneck.")]
        [SerializeField] private bool useCpuSimulation = true;

        // GPU Buffers
        private GraphicsBuffer _voxelBufferA;
        private GraphicsBuffer _voxelBufferB;
        private GraphicsBuffer _svoBuffer;           // Hierarchical occupancy (SVO mip chain)
        private GraphicsBuffer _brickDirtyFlagsBuffer; // Per-brick dirty flags
        private bool _pingPong;

        // CPU-side NativeArray double buffers (used when useCpuSimulation is true)
        private NativeArray<uint> _cpuVoxelA;
        private NativeArray<uint> _cpuVoxelB;
        private NativeArray<uint> _cpuBrickDirtyFlags;
        private NativeArray<uint> _cpuSvoBuffer;
        private bool _cpuBufferDirty;

        // Async job scheduling — schedule one frame, complete next frame
        private JobHandle _cpuSimJobHandle;
        private bool _cpuSimInFlight;

        // SVO hierarchy info
        private int _svoLevelCount;
        private int[] _svoLevelOffsets;  // offset into _svoBuffer for each level
        private int[] _svoGridSizes;     // grid dimension at each level
        private int _svoTotalEntries;
        private bool _worldDirty;        // global flag: need brick map + SVO rebuild

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
        private int _brickMapDirtyKernel;
        private int _markAllDirtyKernel;
        private int _svoBuildLevelKernel;
        private int _svoClearDirtyKernel;
        private int _heatLightKernel;
        private int _lightOnlyKernel;
        private int _copyKernel;
        private bool _kernelsCached;

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

        // Batched write queue: CPU-side writes are collected and uploaded
        // to the GPU once per frame before simulation reads the buffer.
        // Avoids per-voxel SetData overhead and mid-frame GPU stalls.
        private readonly List<VoxelWrite> _pendingWrites = new List<VoxelWrite>(256);
        private uint[] _pendingWriteValues;

        // Properties for external access
        public int WorldSize => worldSize;
        public int BrickSize => brickSize;
        public float VoxelScale => voxelScale;
        public int BrickMapSize => worldSize / brickSize;
        public int TotalVoxels => worldSize * worldSize * worldSize;
        public GraphicsBuffer ReadBuffer => _pingPong ? _voxelBufferB : _voxelBufferA;
        public GraphicsBuffer WriteBuffer => _pingPong ? _voxelBufferA : _voxelBufferB;
        public GraphicsBuffer SVOBuffer => _svoBuffer;
        public Vector3 WorldOrigin => transform.position;
        public float WorldExtent => worldSize * voxelScale;
        public VoxelIndirectInstanceRenderer IndirectInstanceRenderer => indirectInstanceRenderer;
        public bool UseCpuSimulation => useCpuSimulation;

        /// <summary>
        /// Provides direct read-only access to the CPU-side voxel NativeArray.
        /// Only available when useCpuSimulation is true. Avoids GPU readback overhead.
        /// </summary>
        public bool TryGetCpuVoxelNativeData(out NativeArray<uint> data)
        {
            if (useCpuSimulation && _cpuVoxelA.IsCreated)
            {
                // Must complete in-flight sim before external code reads buffer A
                CompleteCpuSimulationIfNeeded();
                data = _cpuVoxelA;
                return true;
            }
            data = default;
            return false;
        }

        // Shader property IDs (cached)
        private static readonly int PropVoxelBuffer = Shader.PropertyToID("_VoxelBuffer");
        private static readonly int PropBrickMap = Shader.PropertyToID("_BrickMap");
        private static readonly int PropSVOBuffer = Shader.PropertyToID("_SVOBuffer");
        private static readonly int PropBrickDirtyFlags = Shader.PropertyToID("_BrickDirtyFlags");
        private static readonly int PropSVOLevel0Offset = Shader.PropertyToID("_SVOLevel0Offset");
        private static readonly int PropSVOLevelOffsets = Shader.PropertyToID("_SVOLevelOffsets");
        private static readonly int PropSVOLevelOffsets2 = Shader.PropertyToID("_SVOLevelOffsets2");
        private static readonly int PropSVOLevelCount = Shader.PropertyToID("_SVOLevelCount");
        private static readonly int PropSrcLevelOffset = Shader.PropertyToID("_SrcLevelOffset");
        private static readonly int PropDstLevelOffset = Shader.PropertyToID("_DstLevelOffset");
        private static readonly int PropSrcGridSize = Shader.PropertyToID("_SrcGridSize");
        private static readonly int PropDstGridSize = Shader.PropertyToID("_DstGridSize");
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

            // Ensure CPU buffers exist when toggling CPU sim at runtime
            if (useCpuSimulation && !_cpuVoxelA.IsCreated)
            {
                InitializeCpuBuffers();
                SyncGpuToCpuBuffers();
            }

            if (useCpuSimulation)
            {
                // =========================================================
                // ASYNC CPU Simulation Flow
                // Schedule jobs one frame, complete the next → main thread
                // is free to do rendering/input while workers run.
                // =========================================================

                // Step 1: Complete previous simulation (workers likely already done → ~0ms)
                CompleteCpuSimulationIfNeeded();

                // Step 2: Flush pending writes (safe: no jobs are in flight)
                FlushPendingWritesCpu();

                // Step 3: Update brick map + SVO (tiny cost, 0.0% in profiler)
                if (_worldDirty)
                {
                    UpdateBrickMapAndSVOCpu(fullRebuild: false);
                    _worldDirty = false;
                }

                // Step 4: Upload to GPU for THIS frame's rendering
                if (_cpuBufferDirty)
                {
                    UploadCpuBuffersToGpu();
                    _cpuBufferDirty = false;
                }

                // Step 5: Schedule next sim (workers run during render + VSync)
                if (enableSimulation)
                {
                    float tickInterval = 1f / Mathf.Max(1f, simulationTickRate);
                    _simulationAccumulator += Time.deltaTime;

                    if (_simulationAccumulator >= tickInterval && !_cpuSimInFlight)
                    {
                        _cpuSimJobHandle = ScheduleCpuSimulationJobs();
                        _cpuSimInFlight = true;
                        _simulationAccumulator -= tickInterval;
                        // Prevent accumulator build-up when running slow
                        if (_simulationAccumulator > tickInterval * 2f)
                            _simulationAccumulator = tickInterval;
                    }
                }
            }
            else
            {
                // =========================================================
                // GPU Simulation Flow (original synchronous path)
                // =========================================================
                FlushPendingWrites();

                if (enableSimulation)
                {
                    bool simulated = RunScheduledSimulation();
                    if (simulated)
                        _worldDirty = true;
                }

                if (_worldDirty)
                {
                    UpdateBrickMapAndSVO(fullRebuild: false);
                    _worldDirty = false;
                }

                UpdateCpuReadbackCache();
            }

            // Update rendering properties
            UpdateRenderProperties();
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
        // Batched Write Upload
        // =====================================================================

        /// <summary>
        /// Uploads all queued voxel modifications to the GPU in one pass.
        /// Sorts writes by buffer index and uploads contiguous runs to
        /// minimise the number of SetData calls. Called once per frame
        /// before simulation dispatches so CPU writes are visible to the
        /// GPU compute shaders without per-voxel upload overhead.
        /// </summary>
        private void FlushPendingWrites()
        {
            int count = _pendingWrites.Count;
            if (count == 0 || ReadBuffer == null) return;

            // Sort by index for contiguous-run detection
            _pendingWrites.Sort((a, b) => a.index.CompareTo(b.index));

            // Reuse values array to avoid per-frame allocation
            if (_pendingWriteValues == null || _pendingWriteValues.Length < count)
                _pendingWriteValues = new uint[Mathf.Max(count, 256)];

            for (int i = 0; i < count; i++)
                _pendingWriteValues[i] = _pendingWrites[i].value;

            // Upload contiguous runs in as few SetData calls as possible
            int runStart = 0;
            while (runStart < count)
            {
                int runEnd = runStart + 1;
                while (runEnd < count &&
                       _pendingWrites[runEnd].index == _pendingWrites[runEnd - 1].index + 1)
                    runEnd++;

                int gpuStart = _pendingWrites[runStart].index;
                int runLength = runEnd - runStart;
                ReadBuffer.SetData(_pendingWriteValues, runStart, gpuStart, runLength);
                runStart = runEnd;
            }

            // Mark dirty flags for affected bricks (CPU-side)
            MarkDirtyBricksForWrites();

            _pendingWrites.Clear();
        }

        /// <summary>
        /// Marks brick dirty flags for all pending writes. Called before clearing
        /// the write queue so we know which bricks need SVO updates.
        /// </summary>
        private void MarkDirtyBricksForWrites()
        {
            if (_brickDirtyFlagsBuffer == null || _pendingWrites.Count == 0) return;

            // Collect unique brick indices
            var dirtyBricks = new HashSet<int>();
            int bms = BrickMapSize;
            foreach (var w in _pendingWrites)
            {
                Vector3Int vpos = VoxelData.Unflatten3D(w.index, worldSize);
                Vector3Int bpos = new Vector3Int(vpos.x / brickSize, vpos.y / brickSize, vpos.z / brickSize);
                if (bpos.x >= 0 && bpos.x < bms && bpos.y >= 0 && bpos.y < bms && bpos.z >= 0 && bpos.z < bms)
                    dirtyBricks.Add(VoxelData.Flatten3D(bpos, bms));
            }

            if (dirtyBricks.Count == 0) return;

            // Upload dirty flags
            var ones = new uint[dirtyBricks.Count];
            var indices = new int[dirtyBricks.Count];
            int i = 0;
            foreach (var bi in dirtyBricks)
            {
                ones[i] = 1u;
                indices[i] = bi;
                i++;
            }

            // Set each dirty brick individually (small count, no contiguous guarantee)
            foreach (var bi in dirtyBricks)
            {
                _brickDirtyFlagsBuffer.SetData(new uint[] { 1u }, 0, bi, 1);
            }

            _worldDirty = true;
        }

        // =====================================================================
        // Initialization
        // =====================================================================

        private void Initialize()
        {
            CreateBuffers();
            CacheKernelIDs();
            CreateRenderingComponents();
            GenerateTerrain();

            if (useCpuSimulation)
            {
                InitializeCpuBuffers();
                SyncGpuToCpuBuffers();
                MarkAllBricksDirtyCpu();
                UpdateBrickMapAndSVOCpu(fullRebuild: true);
                UploadCpuBuffersToGpu();
            }
            else
            {
                MarkAllBricksDirty();
                UpdateBrickMapAndSVO(fullRebuild: true);
            }

            UpdateRenderProperties();
        }

        private void CacheKernelIDs()
        {
            if (_kernelsCached) return;

            if (simulationShader != null)
            {
                _simClearKernel = simulationShader.FindKernel("ClearWriteBuffer");
                _simKernel = simulationShader.FindKernel("SimulateVoxels");
                _heatLightKernel = simulationShader.FindKernel("PropagateHeatAndLight");
                _lightOnlyKernel = simulationShader.FindKernel("PropagateLightOnly");
                _copyKernel = simulationShader.FindKernel("CopyBuffer");
            }

            if (brickMapShader != null)
            {
                _brickMapKernel = brickMapShader.FindKernel("UpdateBrickMap");
                _brickMapDirtyKernel = brickMapShader.FindKernel("UpdateBrickMapDirtyOnly");
                _markAllDirtyKernel = brickMapShader.FindKernel("MarkAllBricksDirty");
            }

            if (svoBuildShader != null)
            {
                _svoBuildLevelKernel = svoBuildShader.FindKernel("BuildSVOLevel");
                _svoClearDirtyKernel = svoBuildShader.FindKernel("ClearDirtyFlags");
            }

            _kernelsCached = true;
        }

        private void CreateBuffers()
        {
            int total = TotalVoxels;
            int bms = BrickMapSize;
            int brickTotal = bms * bms * bms;

            _voxelBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total, sizeof(uint));
            _voxelBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total, sizeof(uint));

            // --- Compute SVO level offsets ---
            // Level 0 = brick map (bms³), level 1 = (bms/2)³, ... until grid size < 2
            var offsets = new System.Collections.Generic.List<int>();
            var gridSizes = new System.Collections.Generic.List<int>();
            int offset = 0;
            int gs = bms;
            while (gs >= 2)
            {
                offsets.Add(offset);
                gridSizes.Add(gs);
                offset += gs * gs * gs;
                gs /= 2;
            }
            _svoLevelCount = offsets.Count;
            _svoLevelOffsets = offsets.ToArray();
            _svoGridSizes = gridSizes.ToArray();
            _svoTotalEntries = offset;

            _svoBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, _svoTotalEntries), sizeof(uint));
            _brickDirtyFlagsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, brickTotal, sizeof(uint));

            // Clear SVO and dirty flags
            var clearSVO = new uint[_svoTotalEntries];
            _svoBuffer.SetData(clearSVO);
            var clearDirty = new uint[brickTotal];
            _brickDirtyFlagsBuffer.SetData(clearDirty);

            Debug.Log($"[VoxelWorld] Created buffers: {worldSize}³ = {total:N0} voxels ({total * 4 / 1024f / 1024f:F1} MB per buffer)");
            Debug.Log($"[VoxelWorld] SVO hierarchy: {_svoLevelCount} levels, {_svoTotalEntries} total entries");
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

            // Copy A→B entirely on GPU using CopyBuffer compute kernel.
            // This avoids an 8 MB GPU→CPU→GPU round-trip that would
            // stall the CPU while waiting for the GPU readback to finish.
            if (simulationShader != null)
            {
                simulationShader.SetBuffer(_copyKernel, PropReadBuffer, _voxelBufferA);
                simulationShader.SetBuffer(_copyKernel, PropWriteBuffer, _voxelBufferB);
                simulationShader.SetInt(PropWorldSize, worldSize);
                int copyGroups = Mathf.CeilToInt(TotalVoxels / 256f);
                simulationShader.Dispatch(_copyKernel, copyGroups, 1, 1);
            }
            else
            {
                // Fallback: CPU round-trip when simulation shader is absent.
                var tempData = new uint[TotalVoxels];
                _voxelBufferA.GetData(tempData);
                _voxelBufferB.SetData(tempData);
            }

            Debug.Log("[VoxelWorld] Terrain generated successfully");
        }

        // =====================================================================
        // Simulation
        // =====================================================================

        private void RunSimulationStep()
        {
            // Note: CPU path uses async scheduling directly from Update().
            // This method is only called for GPU path via RunScheduledSimulation.
            RunGpuSimulationStep();
        }

        private void RunGpuSimulationStep()
        {
            if (simulationShader == null) return;
            CacheKernelIDs();

            var readBuf = ReadBuffer;
            var writeBuf = WriteBuffer;
            int simGroups = Mathf.CeilToInt(worldSize / 4f);

            // Phase 1: Clear write buffer
            simulationShader.SetBuffer(_simClearKernel, PropWriteBuffer, writeBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            int clearGroups = Mathf.CeilToInt(TotalVoxels / 256f);
            simulationShader.Dispatch(_simClearKernel, clearGroups, 1, 1);

            // Phase 2: Simulate (movement + chemical reactions)
            simulationShader.SetBuffer(_simKernel, PropReadBuffer, readBuf);
            simulationShader.SetBuffer(_simKernel, PropWriteBuffer, writeBuf);
            simulationShader.SetBuffer(_simKernel, PropBrickDirtyFlags, _brickDirtyFlagsBuffer);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.SetInt(PropBrickSize, brickSize);
            simulationShader.SetInt(PropBrickMapSize, BrickMapSize);
            simulationShader.SetInt(PropFrameCount, (int)_frameCount);
            simulationShader.SetInt(PropSimStep, (int)_simStep);
            simulationShader.Dispatch(_simKernel, simGroups, simGroups, simGroups);

            // Phase 3: Heat + Light + State changes (writeBuf -> readBuf)
            // The kernel writes every position (air=0), so no separate clear needed.
            simulationShader.SetBuffer(_heatLightKernel, PropReadBuffer, writeBuf);
            simulationShader.SetBuffer(_heatLightKernel, PropWriteBuffer, readBuf);
            simulationShader.SetBuffer(_heatLightKernel, PropBrickDirtyFlags, _brickDirtyFlagsBuffer);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.SetInt(PropBrickSize, brickSize);
            simulationShader.SetInt(PropBrickMapSize, BrickMapSize);
            simulationShader.SetInt(PropFrameCount, (int)_frameCount);
            ApplyFireSimulationTuning(_heatLightKernel);
            simulationShader.Dispatch(_heatLightKernel, simGroups, simGroups, simGroups);

            // Phase 4: Multi-pass light-only propagation for faster convergence.
            // Reduces flickering by allowing light to propagate multiple voxels per tick.
            // After Phase 3, result is in readBuf. Extra passes ping-pong between buffers.
            // Must use an even number of extra passes so final result stays in readBuf.
            int extraPasses = Mathf.Max(0, lightPropagationPasses - 1);
            if (extraPasses % 2 != 0) extraPasses++;

            for (int pass = 0; pass < extraPasses; pass++)
            {
                bool readFromRead = (pass % 2 == 0);
                var src = readFromRead ? readBuf : writeBuf;
                var dst = readFromRead ? writeBuf : readBuf;

                simulationShader.SetBuffer(_lightOnlyKernel, PropReadBuffer, src);
                simulationShader.SetBuffer(_lightOnlyKernel, PropWriteBuffer, dst);
                simulationShader.Dispatch(_lightOnlyKernel, simGroups, simGroups, simGroups);
            }

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
        // Brick Map + SVO Hierarchy
        // =====================================================================

        /// <summary>
        /// Mark all bricks as dirty (used after terrain generation or bulk edits).
        /// </summary>
        private void MarkAllBricksDirty()
        {
            if (brickMapShader == null || _brickDirtyFlagsBuffer == null) return;
            CacheKernelIDs();

            brickMapShader.SetBuffer(_markAllDirtyKernel, PropBrickDirtyFlags, _brickDirtyFlagsBuffer);
            brickMapShader.SetInt(PropBrickMapSize, BrickMapSize);
            int groups = Mathf.CeilToInt(BrickMapSize * BrickMapSize * BrickMapSize / 256f);
            brickMapShader.Dispatch(_markAllDirtyKernel, groups, 1, 1);
            _worldDirty = true;
        }

        /// <summary>
        /// Updates the brick map (SVO level 0) and rebuilds the SVO upper levels.
        /// If fullRebuild is true, processes ALL bricks. Otherwise, only dirty bricks.
        /// </summary>
        private void UpdateBrickMapAndSVO(bool fullRebuild)
        {
            if (brickMapShader == null || _svoBuffer == null) return;
            CacheKernelIDs();

            int bms = BrickMapSize;

            if (fullRebuild)
            {
                // Process all bricks unconditionally
                brickMapShader.SetBuffer(_brickMapKernel, PropVoxelBuffer, ReadBuffer);
                brickMapShader.SetBuffer(_brickMapKernel, PropSVOBuffer, _svoBuffer);
                brickMapShader.SetBuffer(_brickMapKernel, PropBrickDirtyFlags, _brickDirtyFlagsBuffer);
                brickMapShader.SetInt(PropWorldSize, worldSize);
                brickMapShader.SetInt(PropBrickSize, brickSize);
                brickMapShader.SetInt(PropBrickMapSize, bms);
                brickMapShader.SetInt(PropSVOLevel0Offset, _svoLevelOffsets[0]);
                brickMapShader.Dispatch(_brickMapKernel, bms, bms, bms);
            }
            else
            {
                // Only process dirty bricks (dirty-only kernel)
                brickMapShader.SetBuffer(_brickMapDirtyKernel, PropVoxelBuffer, ReadBuffer);
                brickMapShader.SetBuffer(_brickMapDirtyKernel, PropSVOBuffer, _svoBuffer);
                brickMapShader.SetBuffer(_brickMapDirtyKernel, PropBrickDirtyFlags, _brickDirtyFlagsBuffer);
                brickMapShader.SetInt(PropWorldSize, worldSize);
                brickMapShader.SetInt(PropBrickSize, brickSize);
                brickMapShader.SetInt(PropBrickMapSize, bms);
                brickMapShader.SetInt(PropSVOLevel0Offset, _svoLevelOffsets[0]);
                brickMapShader.Dispatch(_brickMapDirtyKernel, bms, bms, bms);
            }

            // Build SVO upper levels (level 1, 2, 3, ...) from level 0
            BuildSVOUpperLevels();
        }

        /// <summary>
        /// Builds SVO upper levels by reducing each level from the one below.
        /// Upper levels are tiny so always fully rebuilt.
        /// </summary>
        private void BuildSVOUpperLevels()
        {
            if (svoBuildShader == null || _svoBuffer == null) return;
            CacheKernelIDs();

            for (int level = 1; level < _svoLevelCount; level++)
            {
                int srcGridSize = _svoGridSizes[level - 1];
                int dstGridSize = _svoGridSizes[level];

                svoBuildShader.SetBuffer(_svoBuildLevelKernel, PropSVOBuffer, _svoBuffer);
                svoBuildShader.SetInt(PropSrcLevelOffset, _svoLevelOffsets[level - 1]);
                svoBuildShader.SetInt(PropDstLevelOffset, _svoLevelOffsets[level]);
                svoBuildShader.SetInt(PropSrcGridSize, srcGridSize);
                svoBuildShader.SetInt(PropDstGridSize, dstGridSize);

                int groups = Mathf.CeilToInt(dstGridSize / 4f);
                svoBuildShader.Dispatch(_svoBuildLevelKernel, groups, groups, groups);
            }
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
            _rayMarchMaterial.SetBuffer(PropSVOBuffer, _svoBuffer);
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

            // Set SVO level offsets for shader
            _rayMarchMaterial.SetVector(PropSVOLevelOffsets, new Vector4(
                _svoLevelCount > 0 ? _svoLevelOffsets[0] : 0,
                _svoLevelCount > 1 ? _svoLevelOffsets[1] : 0,
                _svoLevelCount > 2 ? _svoLevelOffsets[2] : 0,
                _svoLevelCount > 3 ? _svoLevelOffsets[3] : 0
            ));
            _rayMarchMaterial.SetVector(PropSVOLevelOffsets2, new Vector4(
                _svoLevelCount > 4 ? _svoLevelOffsets[4] : 0,
                _svoLevelCount > 5 ? _svoLevelOffsets[5] : 0,
                0, 0
            ));
            _rayMarchMaterial.SetInt(PropSVOLevelCount, _svoLevelCount);

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
            _pendingWrites.Add(new VoxelWrite { index = idx, value = voxelData });
        }

        /// <summary>
        /// Set a sphere of voxels to a given material.
        /// </summary>
        public void SetVoxelSphere(Vector3 center, float radius, uint materialId)
        {
            int r = Mathf.CeilToInt(radius);
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
                _pendingWrites.Add(new VoxelWrite { index = idx, value = voxelValue });
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
            // Fast path: CPU sim has authoritative data in NativeArray
            if (useCpuSimulation && _cpuVoxelA.IsCreated)
            {
                if (_cpuReadbackCache == null || _cpuReadbackCache.Length != TotalVoxels)
                    _cpuReadbackCache = new uint[TotalVoxels];

                // Non-blocking path: if jobs are in flight, keep using last completed snapshot
                // to avoid forcing JobHandle.Complete() from interaction/raycast calls.
                if (_cpuSimInFlight)
                    return _cpuReadbackReady ? _cpuReadbackCache : null;

                if (forceRefresh || !_cpuReadbackReady)
                {
                    _cpuVoxelA.CopyTo(_cpuReadbackCache);
                    _cpuReadbackReady = true;
                }

                return _cpuReadbackCache;
            }

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
            _svoBuffer?.Release();
            _brickDirtyFlagsBuffer?.Release();
            _voxelBufferA = null;
            _voxelBufferB = null;
            _svoBuffer = null;
            _brickDirtyFlagsBuffer = null;
            _cpuReadbackCache = null;
            _pendingWrites.Clear();
            _pendingWriteValues = null;
            _kernelsCached = false;
            _worldDirty = false;

            // Complete any in-flight async jobs before disposing buffers
            if (_cpuSimInFlight)
            {
                _cpuSimJobHandle.Complete();
                _cpuSimInFlight = false;
            }

            DisposeCpuBuffers();

            if (_rayMarchMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(_rayMarchMaterial);
                else
                    DestroyImmediate(_rayMarchMaterial);
            }
        }

        // =====================================================================
        // CPU Simulation (Jobs + Burst)
        // =====================================================================

        private void InitializeCpuBuffers()
        {
            int total = TotalVoxels;
            int brickTotal = BrickMapSize * BrickMapSize * BrickMapSize;

            _cpuVoxelA = new NativeArray<uint>(total, Allocator.Persistent);
            _cpuVoxelB = new NativeArray<uint>(total, Allocator.Persistent);
            _cpuBrickDirtyFlags = new NativeArray<uint>(brickTotal, Allocator.Persistent);
            _cpuSvoBuffer = new NativeArray<uint>(Mathf.Max(1, _svoTotalEntries), Allocator.Persistent);
            _cpuBufferDirty = false;

            Debug.Log("[VoxelWorld] CPU simulation buffers created (Jobs + Burst mode)");
        }

        private void DisposeCpuBuffers()
        {
            if (_cpuVoxelA.IsCreated) _cpuVoxelA.Dispose();
            if (_cpuVoxelB.IsCreated) _cpuVoxelB.Dispose();
            if (_cpuBrickDirtyFlags.IsCreated) _cpuBrickDirtyFlags.Dispose();
            if (_cpuSvoBuffer.IsCreated) _cpuSvoBuffer.Dispose();
        }

        /// <summary>
        /// One-time download of GPU terrain data → CPU NativeArrays after terrain generation.
        /// </summary>
        private void SyncGpuToCpuBuffers()
        {
            if (!_cpuVoxelA.IsCreated) return;

            var temp = new uint[TotalVoxels];
            _voxelBufferA.GetData(temp);
            _cpuVoxelA.CopyFrom(temp);
            _cpuVoxelB.CopyFrom(temp);
            _cpuBufferDirty = false;
        }

        /// <summary>
        /// CPU-side pending write flush: writes directly to NativeArray
        /// (no sorting or batching needed — random access is cheap on CPU).
        /// </summary>
        private void FlushPendingWritesCpu()
        {
            int count = _pendingWrites.Count;
            if (count == 0 || !_cpuVoxelA.IsCreated) return;

            int bms = BrickMapSize;
            for (int i = 0; i < count; i++)
            {
                var w = _pendingWrites[i];
                _cpuVoxelA[w.index] = w.value;

                // Mark brick dirty
                Vector3Int vpos = VoxelData.Unflatten3D(w.index, worldSize);
                int bx = vpos.x / brickSize;
                int by = vpos.y / brickSize;
                int bz = vpos.z / brickSize;
                if (bx >= 0 && bx < bms && by >= 0 && by < bms && bz >= 0 && bz < bms)
                {
                    int brickIdx = VoxelData.Flatten3D(bx, by, bz, bms);
                    _cpuBrickDirtyFlags[brickIdx] = 1u;
                }
            }

            _pendingWrites.Clear();
            _worldDirty = true;
            _cpuBufferDirty = true;
            _cpuReadbackReady = false;
        }

        /// <summary>
        /// Complete any in-flight async CPU simulation.
        /// Called at frame start — workers typically already finished → near-instant.
        /// </summary>
        private void CompleteCpuSimulationIfNeeded()
        {
            if (!_cpuSimInFlight) return;

            _cpuSimJobHandle.Complete();
            _cpuSimInFlight = false;
            _simStep++;
            _worldDirty = true;
            _cpuBufferDirty = true;
            _cpuReadbackReady = false;
        }

        /// <summary>
        /// Schedule one full simulation tick as async Burst jobs — returns immediately.
        /// Workers execute during rendering + VSync. Complete next frame via
        /// CompleteCpuSimulationIfNeeded().
        /// Phase 1: Clear B → Phase 2: Simulate A→B → Phase 3: HeatLight B→A
        /// → Phase 4: LightOnly ping-pong (even passes, result stays in A).
        /// </summary>
        private unsafe JobHandle ScheduleCpuSimulationJobs()
        {
            if (!_cpuVoxelA.IsCreated || !_cpuVoxelB.IsCreated) return default;

            int total = TotalVoxels;

            // Phase 1: Clear write buffer (B)
            var clearJob = new ClearBufferJob { buffer = _cpuVoxelB };
            JobHandle clearHandle = clearJob.Schedule(total, 2048);

            // Phase 2: Simulate voxels (read A → write B via CAS)
            var simJob = new SimulateVoxelsJob
            {
                readBuffer = _cpuVoxelA,
                writeBufferPtr = (int*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(_cpuVoxelB),
                dirtyFlagsPtr = (uint*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(_cpuBrickDirtyFlags),
                worldSize = worldSize,
                brickSize = brickSize,
                brickMapSize = BrickMapSize,
                frameCount = _frameCount,
                simStep = _simStep
            };
            JobHandle simHandle = simJob.Schedule(total, 512, clearHandle);

            // Phase 3: Heat + Light propagation (read B → write A)
            GetFireParams(out int fSpreadTemp, out int fWoodTemp, out int fLeafTemp,
                out int fCoalTemp, out int fSnowTemp, out int fWaterTemp,
                out int fBurnLightTemp, out int fRiseRate, out int fCoolRate,
                out int fSinkCool, out int fBurnoutDiv);

            var heatLightJob = new PropagateHeatAndLightJob
            {
                readBuffer = _cpuVoxelB,
                writeBuffer = _cpuVoxelA,
                dirtyFlags = _cpuBrickDirtyFlags,
                worldSize = worldSize,
                brickSize = brickSize,
                brickMapSize = BrickMapSize,
                frameCount = _frameCount,
                fireSpreadNeighborTemp = fSpreadTemp,
                woodCharTemp = fWoodTemp,
                leafBurnTemp = fLeafTemp,
                coalBurnoutTemp = fCoalTemp,
                snowMeltTemp = fSnowTemp,
                waterEvapTemp = fWaterTemp,
                burningLightTemp = fBurnLightTemp,
                heatRiseRate = fRiseRate,
                coolRate = fCoolRate,
                heatSinkExtraCool = fSinkCool,
                coalBurnoutChanceDiv = fBurnoutDiv
            };
            JobHandle heatHandle = heatLightJob.Schedule(total, 1024, simHandle);

            // Phase 4: Multi-pass light-only propagation (even number of passes)
            // After Phase 3, result is in A. Even passes keep result in A.
            int extraPasses = Mathf.Max(0, lightPropagationPasses - 1);
            if (extraPasses % 2 != 0) extraPasses++;

            JobHandle lastHandle = heatHandle;
            for (int pass = 0; pass < extraPasses; pass++)
            {
                bool readFromA = (pass % 2 == 0);
                var lightJob = new PropagateLightOnlyJob
                {
                    readBuffer = readFromA ? _cpuVoxelA : _cpuVoxelB,
                    writeBuffer = readFromA ? _cpuVoxelB : _cpuVoxelA,
                    worldSize = worldSize,
                    burningLightTemp = fBurnLightTemp
                };
                lastHandle = lightJob.Schedule(total, 1024, lastHandle);
            }

            // Return handle — do NOT call Complete(). Workers run asynchronously.
            return lastHandle;
        }

        /// <summary>
        /// Extract resolved fire simulation parameters based on current profile.
        /// </summary>
        private void GetFireParams(out int spreadTemp, out int woodTemp, out int leafTemp,
            out int coalTemp, out int snowTemp, out int waterTemp, out int burnLightTemp,
            out int riseRate, out int decayRate, out int sinkCool, out int burnoutDiv)
        {
            spreadTemp = fireSpreadNeighborTemp;
            woodTemp = woodCharTemp;
            leafTemp = leafBurnTemp;
            coalTemp = coalBurnoutTemp;
            snowTemp = snowMeltTemp;
            waterTemp = waterEvapTemp;
            burnLightTemp = burningLightTemp;
            riseRate = heatRiseRate;
            decayRate = coolRate;
            sinkCool = heatSinkExtraCool;
            burnoutDiv = coalBurnoutChanceDiv;

            if (fireProfile == FireSimulationProfile.Fast)
            {
                spreadTemp = 7; woodTemp = 10; leafTemp = 8; coalTemp = 13;
                snowTemp = 3; waterTemp = 10; burnLightTemp = 8;
                riseRate = 2; decayRate = 1; sinkCool = 0; burnoutDiv = 5;
            }
            else if (fireProfile == FireSimulationProfile.Realistic)
            {
                spreadTemp = 10; woodTemp = 13; leafTemp = 10; coalTemp = 15;
                snowTemp = 5; waterTemp = 13; burnLightTemp = 10;
                riseRate = 1; decayRate = 1; sinkCool = 1; burnoutDiv = 12;
            }

            spreadTemp = Mathf.Clamp(spreadTemp, 4, 15);
            woodTemp = Mathf.Clamp(woodTemp, 6, 15);
            leafTemp = Mathf.Clamp(leafTemp, 6, 15);
            coalTemp = Mathf.Clamp(coalTemp, 8, 15);
            snowTemp = Mathf.Clamp(snowTemp, 2, 8);
            waterTemp = Mathf.Clamp(waterTemp, 8, 15);
            burnLightTemp = Mathf.Clamp(burnLightTemp, 6, 15);
            riseRate = Mathf.Clamp(riseRate, 1, 3);
            decayRate = Mathf.Clamp(decayRate, 1, 3);
            sinkCool = Mathf.Clamp(sinkCool, 0, 2);
            burnoutDiv = Mathf.Max(2, burnoutDiv);
        }

        /// <summary>
        /// CPU-side brick map and SVO hierarchy update (replaces GPU compute).
        /// </summary>
        private void UpdateBrickMapAndSVOCpu(bool fullRebuild)
        {
            if (!_cpuVoxelA.IsCreated || !_cpuSvoBuffer.IsCreated) return;

            int brickCount = BrickMapSize * BrickMapSize * BrickMapSize;

            // Update brick map (SVO level 0)
            var brickJob = new UpdateBrickMapJob
            {
                voxelBuffer = _cpuVoxelA,
                svoBuffer = _cpuSvoBuffer,
                dirtyFlags = _cpuBrickDirtyFlags,
                worldSize = worldSize,
                brickSize = brickSize,
                brickMapSize = BrickMapSize,
                svoLevel0Offset = _svoLevelOffsets[0],
                fullRebuild = fullRebuild
            };
            JobHandle brickHandle = brickJob.Schedule(brickCount, 64);

            // Build SVO upper levels (level 1, 2, 3, ...)
            JobHandle lastHandle = brickHandle;
            for (int level = 1; level < _svoLevelCount; level++)
            {
                int dstTotal = _svoGridSizes[level] * _svoGridSizes[level] * _svoGridSizes[level];
                var svoJob = new BuildSVOLevelJob
                {
                    svoBuffer = _cpuSvoBuffer,
                    srcLevelOffset = _svoLevelOffsets[level - 1],
                    dstLevelOffset = _svoLevelOffsets[level],
                    srcGridSize = _svoGridSizes[level - 1],
                    dstGridSize = _svoGridSizes[level]
                };
                lastHandle = svoJob.Schedule(dstTotal, 64, lastHandle);
            }

            lastHandle.Complete();
            _cpuBufferDirty = true;
        }

        /// <summary>
        /// Mark all bricks as dirty on CPU NativeArray.
        /// </summary>
        private void MarkAllBricksDirtyCpu()
        {
            if (!_cpuBrickDirtyFlags.IsCreated) return;
            var markJob = new MarkAllDirtyJob { dirtyFlags = _cpuBrickDirtyFlags };
            markJob.Schedule(_cpuBrickDirtyFlags.Length, 256).Complete();
            _worldDirty = true;
        }

        /// <summary>
        /// Upload CPU NativeArray data → GPU GraphicsBuffers for rendering.
        /// Single SetData call per buffer — efficient DMA transfer.
        /// Called once per frame when data has changed.
        /// </summary>
        private void UploadCpuBuffersToGpu()
        {
            if (!_cpuVoxelA.IsCreated) return;

            // Upload voxel data to the non-read buffer first, then flip.
            // This avoids writing into the buffer the GPU may still be reading,
            // which can force a stall and inflate WaitForGPU on D3D12.
            if (ReadBuffer != null && WriteBuffer != null)
            {
                WriteBuffer.SetData(_cpuVoxelA);
                _pingPong = !_pingPong;
            }
            else if (ReadBuffer != null)
            {
                ReadBuffer.SetData(_cpuVoxelA);
            }

            // Upload SVO hierarchy (used by raymarching for accelerated skipping)
            if (_svoBuffer != null && _cpuSvoBuffer.IsCreated)
                _svoBuffer.SetData(_cpuSvoBuffer);
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
