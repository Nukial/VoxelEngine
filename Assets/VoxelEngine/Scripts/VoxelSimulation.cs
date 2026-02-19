using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Fire simulation tuning profile values.
    /// </summary>
    public enum FireSimulationProfile
    {
        Fast,
        Realistic,
        Custom
    }

    /// <summary>
    /// Encapsulates fire simulation tuning parameters exposed in the Inspector.
    /// </summary>
    [System.Serializable]
    public struct FireSimulationSettings
    {
        public FireSimulationProfile profile;
        [Range(4, 14)] public int fireSpreadNeighborTemp;
        [Range(6, 15)] public int woodCharTemp;
        [Range(6, 15)] public int leafBurnTemp;
        [Range(8, 15)] public int coalBurnoutTemp;
        [Range(2, 8)] public int snowMeltTemp;
        [Range(8, 15)] public int waterEvapTemp;
        [Range(6, 15)] public int burningLightTemp;
        [Range(1, 3)] public int heatRiseRate;
        [Range(1, 3)] public int coolRate;
        [Range(0, 2)] public int heatSinkExtraCool;
        [Range(2, 20)] public int coalBurnoutChanceDiv;

        public static FireSimulationSettings Default => new FireSimulationSettings
        {
            profile = FireSimulationProfile.Realistic,
            fireSpreadNeighborTemp = 10,
            woodCharTemp = 13,
            leafBurnTemp = 10,
            coalBurnoutTemp = 15,
            snowMeltTemp = 5,
            waterEvapTemp = 13,
            burningLightTemp = 10,
            heatRiseRate = 1,
            coolRate = 1,
            heatSinkExtraCool = 1,
            coalBurnoutChanceDiv = 12
        };
    }

    /// <summary>
    /// Runs the voxel simulation (movement, heat, light propagation) and brick map updates
    /// via compute shaders each frame.
    /// </summary>
    public class VoxelSimulation
    {
        // Shader property IDs (cached)
        private static readonly int PropReadBuffer = Shader.PropertyToID("_ReadBuffer");
        private static readonly int PropWriteBuffer = Shader.PropertyToID("_WriteBuffer");
        private static readonly int PropVoxelBuffer = Shader.PropertyToID("_VoxelBuffer");
        private static readonly int PropBrickMap = Shader.PropertyToID("_BrickMap");
        private static readonly int PropWorldSize = Shader.PropertyToID("_WorldSize");
        private static readonly int PropBrickSize = Shader.PropertyToID("_BrickSize");
        private static readonly int PropBrickMapSize = Shader.PropertyToID("_BrickMapSize");
        private static readonly int PropFrameCount = Shader.PropertyToID("_FrameCount");
        private static readonly int PropSimStep = Shader.PropertyToID("_SimStep");

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

        // Kernel IDs
        private int _simClearKernel;
        private int _simKernel;
        private int _heatLightKernel;
        private int _lightOnlyKernel;
        private int _brickMapKernel;
        private bool _kernelsCached;

        // Frame counters
        private uint _simStep;

        // Scheduling
        private float _simulationAccumulator;

        public void CacheKernelIDs(ComputeShader simulationShader, ComputeShader brickMapShader)
        {
            if (_kernelsCached) return;

            if (simulationShader != null)
            {
                _simClearKernel = simulationShader.FindKernel("ClearWriteBuffer");
                _simKernel = simulationShader.FindKernel("SimulateVoxels");
                _heatLightKernel = simulationShader.FindKernel("PropagateHeatAndLight");
                _lightOnlyKernel = simulationShader.FindKernel("PropagateLightOnly");
            }

            if (brickMapShader != null)
                _brickMapKernel = brickMapShader.FindKernel("UpdateBrickMap");

            _kernelsCached = true;
        }

        /// <summary>
        /// Runs scheduled simulation steps based on tick rate. Returns true if any step ran.
        /// </summary>
        public bool RunScheduled(ComputeShader simulationShader, ComputeShader brickMapShader,
            VoxelBufferManager buffers, uint frameCount, float simulationTickRate,
            int simulationStepsPerFrame, int lightPropagationPasses,
            FireSimulationSettings fireSettings)
        {
            float tickRate = Mathf.Max(1f, simulationTickRate);
            float tickInterval = 1f / tickRate;
            _simulationAccumulator += Time.deltaTime;

            int maxSteps = Mathf.Max(1, simulationStepsPerFrame);
            bool simulated = false;

            while (_simulationAccumulator >= tickInterval && maxSteps > 0)
            {
                RunStep(simulationShader, buffers, frameCount, lightPropagationPasses, fireSettings);
                _simulationAccumulator -= tickInterval;
                maxSteps--;
                simulated = true;
            }

            if (_simulationAccumulator > tickInterval * 4f)
                _simulationAccumulator = tickInterval;

            return simulated;
        }

        /// <summary>
        /// Execute a single simulation step (clear, simulate, heat/light, light propagation).
        /// </summary>
        public void RunStep(ComputeShader simulationShader, VoxelBufferManager buffers,
            uint frameCount, int lightPropagationPasses, FireSimulationSettings fireSettings)
        {
            if (simulationShader == null) return;

            var readBuf = buffers.ReadBuffer;
            var writeBuf = buffers.WriteBuffer;
            int worldSize = buffers.WorldSize;
            int totalVoxels = buffers.TotalVoxels;
            int simGroups = Mathf.CeilToInt(worldSize / 4f);

            // Phase 1: Clear write buffer
            simulationShader.SetBuffer(_simClearKernel, PropWriteBuffer, writeBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            int clearGroups = Mathf.CeilToInt(totalVoxels / 256f);
            simulationShader.Dispatch(_simClearKernel, clearGroups, 1, 1);

            // Phase 2: Simulate (movement + chemical reactions)
            simulationShader.SetBuffer(_simKernel, PropReadBuffer, readBuf);
            simulationShader.SetBuffer(_simKernel, PropWriteBuffer, writeBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.SetInt(PropFrameCount, (int)frameCount);
            simulationShader.SetInt(PropSimStep, (int)_simStep);
            simulationShader.Dispatch(_simKernel, simGroups, simGroups, simGroups);

            // Phase 3: Heat + Light + State changes (writeBuf -> readBuf)
            simulationShader.SetBuffer(_heatLightKernel, PropReadBuffer, writeBuf);
            simulationShader.SetBuffer(_heatLightKernel, PropWriteBuffer, readBuf);
            simulationShader.SetInt(PropWorldSize, worldSize);
            simulationShader.SetInt(PropFrameCount, (int)frameCount);
            ApplyFireTuning(simulationShader, _heatLightKernel, fireSettings);
            simulationShader.Dispatch(_heatLightKernel, simGroups, simGroups, simGroups);

            // Phase 4: Multi-pass light-only propagation
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

        /// <summary>
        /// Update the brick map from the current read buffer.
        /// </summary>
        public void UpdateBrickMap(ComputeShader brickMapShader, VoxelBufferManager buffers)
        {
            if (brickMapShader == null) return;

            int worldSize = buffers.WorldSize;
            int brickSize = buffers.BrickSize;
            int brickMapSize = buffers.BrickMapSize;

            brickMapShader.SetBuffer(_brickMapKernel, PropVoxelBuffer, buffers.ReadBuffer);
            brickMapShader.SetBuffer(_brickMapKernel, PropBrickMap, buffers.BrickMapBuffer);
            brickMapShader.SetInt(PropWorldSize, worldSize);
            brickMapShader.SetInt(PropBrickSize, brickSize);
            brickMapShader.SetInt(PropBrickMapSize, brickMapSize);

            brickMapShader.Dispatch(_brickMapKernel, brickMapSize, brickMapSize, brickMapSize);
        }

        public void Reset()
        {
            _kernelsCached = false;
            _simulationAccumulator = 0f;
            _simStep = 0;
        }

        private void ApplyFireTuning(ComputeShader shader, int kernel, FireSimulationSettings s)
        {
            int spreadTemp = s.fireSpreadNeighborTemp;
            int woodTemp = s.woodCharTemp;
            int leafTemp = s.leafBurnTemp;
            int coalTemp = s.coalBurnoutTemp;
            int snowTemp = s.snowMeltTemp;
            int waterTemp = s.waterEvapTemp;
            int burnLightTemp = s.burningLightTemp;
            int riseRate = s.heatRiseRate;
            int decayRate = s.coolRate;
            int sinkCool = s.heatSinkExtraCool;
            int burnoutDiv = s.coalBurnoutChanceDiv;

            if (s.profile == FireSimulationProfile.Fast)
            {
                spreadTemp = 7;  woodTemp = 10; leafTemp = 8;
                coalTemp = 13;   snowTemp = 3;  waterTemp = 10;
                burnLightTemp = 8; riseRate = 2; decayRate = 1;
                sinkCool = 0;    burnoutDiv = 5;
            }
            else if (s.profile == FireSimulationProfile.Realistic)
            {
                spreadTemp = 10; woodTemp = 13; leafTemp = 10;
                coalTemp = 15;   snowTemp = 5;  waterTemp = 13;
                burnLightTemp = 10; riseRate = 1; decayRate = 1;
                sinkCool = 1;    burnoutDiv = 12;
            }

            shader.SetInt(PropFireSpreadNeighborTemp, Mathf.Clamp(spreadTemp, 4, 15));
            shader.SetInt(PropWoodCharTemp, Mathf.Clamp(woodTemp, 6, 15));
            shader.SetInt(PropLeafBurnTemp, Mathf.Clamp(leafTemp, 6, 15));
            shader.SetInt(PropCoalBurnoutTemp, Mathf.Clamp(coalTemp, 8, 15));
            shader.SetInt(PropSnowMeltTemp, Mathf.Clamp(snowTemp, 2, 8));
            shader.SetInt(PropWaterEvapTemp, Mathf.Clamp(waterTemp, 8, 15));
            shader.SetInt(PropBurningLightTemp, Mathf.Clamp(burnLightTemp, 6, 15));
            shader.SetInt(PropHeatRiseRate, Mathf.Clamp(riseRate, 1, 3));
            shader.SetInt(PropCoolRate, Mathf.Clamp(decayRate, 1, 3));
            shader.SetInt(PropHeatSinkExtraCool, Mathf.Clamp(sinkCool, 0, 2));
            shader.SetInt(PropCoalBurnoutChanceDiv, Mathf.Max(2, burnoutDiv));
        }
    }
}
