using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Handles GPU-based procedural terrain generation via compute shader.
    /// </summary>
    public class VoxelTerrainGenerator
    {
        // Shader property IDs (cached)
        private static readonly int PropVoxelBuffer = Shader.PropertyToID("_VoxelBuffer");
        private static readonly int PropWorldSize = Shader.PropertyToID("_WorldSize");
        private static readonly int PropSeed = Shader.PropertyToID("_Seed");
        private static readonly int PropTerrainScale = Shader.PropertyToID("_TerrainScale");
        private static readonly int PropCaveScale = Shader.PropertyToID("_CaveScale");
        private static readonly int PropCaveThreshold = Shader.PropertyToID("_CaveThreshold");
        private static readonly int PropWaterLevel = Shader.PropertyToID("_WaterLevel");

        private int _terrainKernel;

        /// <summary>
        /// Generate terrain into bufferA, then copy to bufferB for initial double-buffer state.
        /// </summary>
        public void Generate(ComputeShader terrainGenShader, VoxelBufferManager buffers,
            float seed, float terrainScale, float caveScale, float caveThreshold)
        {
            if (terrainGenShader == null)
            {
                Debug.LogError("[VoxelTerrainGenerator] TerrainGeneration compute shader not assigned!");
                return;
            }

            int worldSize = buffers.WorldSize;
            _terrainKernel = terrainGenShader.FindKernel("GenerateTerrain");

            terrainGenShader.SetBuffer(_terrainKernel, PropVoxelBuffer, buffers.VoxelBufferA);
            terrainGenShader.SetInt(PropWorldSize, worldSize);
            terrainGenShader.SetFloat(PropSeed, seed);
            terrainGenShader.SetFloat(PropTerrainScale, terrainScale);
            terrainGenShader.SetFloat(PropCaveScale, caveScale);
            terrainGenShader.SetFloat(PropCaveThreshold, caveThreshold);
            terrainGenShader.SetInt(PropWaterLevel, Mathf.RoundToInt(worldSize * 0.3f));

            int groups = Mathf.CeilToInt(worldSize / 4f);
            terrainGenShader.Dispatch(_terrainKernel, groups, groups, groups);

            // Copy A to B for initial state
            var tempData = new uint[buffers.TotalVoxels];
            buffers.VoxelBufferA.GetData(tempData);
            buffers.VoxelBufferB.SetData(tempData);

            Debug.Log("[VoxelTerrainGenerator] Terrain generated successfully");
        }
    }
}
