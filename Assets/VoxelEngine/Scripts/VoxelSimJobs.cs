// ============================================================================
// VoxelSimJobs.cs
// CPU-side Burst-compiled jobs that mirror the GPU compute shader simulation.
// Offloads voxel simulation, heat/light propagation, brick map, and SVO
// building from the GPU to CPU worker threads via the Unity Job System.
// ============================================================================

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using System.Threading;

namespace VoxelEngine
{
    // =========================================================================
    // VoxelSimUtil — Burst-compatible static utilities (mirrors VoxelCommon.hlsl)
    // =========================================================================

    public static class VoxelSimUtil
    {
        // Material IDs — must match VoxelCommon.hlsl and VoxelData.cs
        public const uint MAT_AIR   = 0;
        public const uint MAT_STONE = 1;
        public const uint MAT_DIRT  = 2;
        public const uint MAT_GRASS = 3;
        public const uint MAT_SAND  = 4;
        public const uint MAT_WATER = 5;
        public const uint MAT_LAVA  = 6;
        public const uint MAT_WOOD  = 7;
        public const uint MAT_LEAF  = 8;
        public const uint MAT_SNOW  = 9;
        public const uint MAT_GLASS = 10;
        public const uint MAT_IRON  = 11;
        public const uint MAT_COAL  = 12;
        public const uint MAT_GOLD  = 13;
        public const uint MAT_STEAM = 14;

        // Physics categories
        public const uint PHYS_STATIC = 0;
        public const uint PHYS_POWDER = 1;
        public const uint PHYS_LIQUID = 2;
        public const uint PHYS_GAS    = 3;

        // --- Packing / Unpacking ---

        public static uint PackVoxel(uint materialId, uint color565, uint aux)
        {
            return (aux << 24) | (color565 << 8) | (materialId & 0xFFu);
        }

        public static uint GetMaterialId(uint voxel) => voxel & 0xFFu;

        public static uint GetColor565(uint voxel) => (voxel >> 8) & 0xFFFFu;

        public static uint GetAuxData(uint voxel) => (voxel >> 24) & 0xFFu;

        public static uint SetAuxData(uint voxel, uint aux)
        {
            return (voxel & 0x00FFFFFFu) | ((aux & 0xFFu) << 24);
        }

        // --- Temperature & Light (packed in Aux byte bits 0-3 / 4-7) ---

        public static uint GetTemperature(uint voxel) => GetAuxData(voxel) & 0xFu;

        public static uint GetLightLevel(uint voxel) => (GetAuxData(voxel) >> 4) & 0xFu;

        public static uint SetTemperature(uint voxel, uint temp)
        {
            uint aux = GetAuxData(voxel);
            aux = (aux & 0xF0u) | (temp & 0xFu);
            return SetAuxData(voxel, aux);
        }

        public static uint SetLightLevel(uint voxel, uint light)
        {
            uint aux = GetAuxData(voxel);
            aux = (aux & 0x0Fu) | ((light & 0xFu) << 4);
            return SetAuxData(voxel, aux);
        }

        public static uint SetTempAndLight(uint voxel, uint temp, uint light)
        {
            uint aux = (temp & 0xFu) | ((light & 0xFu) << 4);
            return SetAuxData(voxel, aux);
        }

        // --- Material Properties ---

        public static uint GetPhysicsCategory(uint matId)
        {
            if (matId == MAT_SAND || matId == MAT_SNOW || matId == MAT_COAL) return PHYS_POWDER;
            if (matId == MAT_WATER || matId == MAT_LAVA) return PHYS_LIQUID;
            if (matId == MAT_STEAM) return PHYS_GAS;
            return PHYS_STATIC;
        }

        public static bool BlocksLight(uint matId)
        {
            return matId != MAT_AIR && matId != MAT_GLASS && matId != MAT_WATER && matId != MAT_STEAM;
        }

        public static uint GetMaterialHeatSource(uint matId) => matId == MAT_LAVA ? 15u : 0u;

        public static bool IsHeatSink(uint matId)
        {
            return matId == MAT_WATER || matId == MAT_SNOW || matId == MAT_AIR;
        }

        public static uint GetMaterialEmission(uint matId) => matId == MAT_LAVA ? 15u : 0u;

        // --- 3D Indexing ---

        public static int Flatten3D(int3 pos, int size)
        {
            return pos.x + pos.y * size + pos.z * size * size;
        }

        public static int3 Unflatten3D(int index, int size)
        {
            int ss = size * size;
            int z = index / ss;
            int rem = index - z * ss;
            int y = rem / size;
            int x = rem - y * size;
            return new int3(x, y, z);
        }

        public static bool IsInBounds(int3 pos, int size)
        {
            return math.all(pos >= 0) & math.all(pos < size);
        }

        // --- Hashing (mirrors VoxelCommon.hlsl exactly) ---

        public static uint VoxelHash(uint x)
        {
            x = ((x >> 16) ^ x) * 0x45d9f3bu;
            x = ((x >> 16) ^ x) * 0x45d9f3bu;
            x = (x >> 16) ^ x;
            return x;
        }

        public static uint HashPos(int3 pos, uint seed)
        {
            uint h = (uint)pos.x * 73856093u;
            h ^= (uint)pos.y * 19349663u;
            h ^= (uint)pos.z * 83492791u;
            h ^= seed * 1299709u;
            return VoxelHash(h);
        }
    }

    // =========================================================================
    // ClearBufferJob — Zero the write buffer before simulation
    // =========================================================================

    [BurstCompile]
    public struct ClearBufferJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<uint> buffer;
        public void Execute(int index) { buffer[index] = 0u; }
    }

    // =========================================================================
    // CopyBufferJob — GPU-free buffer copy (A → B or B → A)
    // =========================================================================

    [BurstCompile]
    public struct CopyBufferJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> source;
        [WriteOnly] public NativeArray<uint> destination;
        public void Execute(int index) { destination[index] = source[index]; }
    }

    // =========================================================================
    // SimulateVoxelsJob — Cellular automata movement simulation
    // Mirrors SimulateVoxels kernel from VoxelSimulation.compute
    // Uses Interlocked.CompareExchange for CAS conflict resolution
    // =========================================================================

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast)]
    public unsafe struct SimulateVoxelsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> readBuffer;

        // Raw pointer for Interlocked.CompareExchange (CAS) — matches GPU CAS
        [NativeDisableUnsafePtrRestriction] public int* writeBufferPtr;

        // Dirty flags — idempotent writes of 1, safe without atomics
        [NativeDisableUnsafePtrRestriction] public uint* dirtyFlagsPtr;

        public int worldSize;
        public int brickSize;
        public int brickMapSize;
        public uint frameCount;
        public uint simStep;

        public void Execute(int index)
        {
            int3 pos = VoxelSimUtil.Unflatten3D(index, worldSize);

            uint voxel = readBuffer[index];
            uint mat = VoxelSimUtil.GetMaterialId(voxel);

            // Air: do nothing (write buffer already cleared)
            if (mat == VoxelSimUtil.MAT_AIR) return;

            uint physCat = VoxelSimUtil.GetPhysicsCategory(mat);

            // --- Static Materials ---
            if (physCat == VoxelSimUtil.PHYS_STATIC)
            {
                TryClaim(pos, voxel);
                return;
            }

            uint rng = Rand(pos);
            bool moved = false;

            // --- Powder Physics (Sand, Snow, Coal) ---
            if (physCat == VoxelSimUtil.PHYS_POWDER)
            {
                // 1. Fall straight down
                int3 below = pos + new int3(0, -1, 0);
                if (VoxelSimUtil.IsInBounds(below, worldSize) &&
                    ReadMatAt(below) == VoxelSimUtil.MAT_AIR)
                {
                    moved = TryClaim(below, voxel);
                    if (moved) return;
                }

                // 2. Fall through liquid (swap)
                if (!moved && VoxelSimUtil.IsInBounds(below, worldSize))
                {
                    uint belowMat = ReadMatAt(below);
                    if (belowMat == VoxelSimUtil.MAT_WATER || belowMat == VoxelSimUtil.MAT_STEAM)
                    {
                        uint belowVoxel = ReadAt(below);
                        moved = TryClaim(below, voxel);
                        if (moved)
                        {
                            TryClaim(pos, belowVoxel);
                            return;
                        }
                    }
                }

                // 3. Diagonal falls (angle of repose)
                if (!moved)
                {
                    int dx = (rng & 1u) != 0 ? 1 : -1;
                    int dz = ((rng >> 1) & 1u) != 0 ? 1 : -1;

                    if (!moved) { int3 d = pos + new int3(dx, -1, 0);   if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                    if (!moved) { int3 d = pos + new int3(0, -1, dz);   if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                    if (!moved) { int3 d = pos + new int3(-dx, -1, 0);  if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                    if (!moved) { int3 d = pos + new int3(0, -1, -dz);  if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                }

                // 4. Stay in place
                if (!moved) TryClaim(pos, voxel);
                return;
            }

            // --- Liquid Physics (Water, Lava) ---
            if (physCat == VoxelSimUtil.PHYS_LIQUID)
            {
                // Lava chemical interaction (every other frame): lava + water → stone
                if (mat == VoxelSimUtil.MAT_LAVA && (frameCount & 1u) == 0u)
                {
                    if (HasNeighborMat(pos, VoxelSimUtil.MAT_WATER))
                    {
                        uint stoneVoxel = VoxelSimUtil.PackVoxel(VoxelSimUtil.MAT_STONE, 0, 0);
                        TryClaim(pos, stoneVoxel);
                        return;
                    }
                }

                // Water near lava → steam
                if (mat == VoxelSimUtil.MAT_WATER)
                {
                    if (HasNeighborMat(pos, VoxelSimUtil.MAT_LAVA))
                    {
                        uint steamVoxel = VoxelSimUtil.PackVoxel(VoxelSimUtil.MAT_STEAM, 0, 0);
                        TryClaim(pos, steamVoxel);
                        return;
                    }
                }

                // 1. Fall down
                int3 below = pos + new int3(0, -1, 0);
                if (VoxelSimUtil.IsInBounds(below, worldSize) &&
                    ReadMatAt(below) == VoxelSimUtil.MAT_AIR)
                {
                    moved = TryClaim(below, voxel);
                    if (moved) return;
                }

                // 2. Diagonal down
                if (!moved)
                {
                    int dx = (rng & 1u) != 0 ? 1 : -1;
                    int dz = ((rng >> 1) & 1u) != 0 ? 1 : -1;

                    if (!moved) { int3 d = pos + new int3(dx, -1, 0);  if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                    if (!moved) { int3 d = pos + new int3(0, -1, dz);  if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                    if (!moved) { int3 d = pos + new int3(-dx, -1, 0); if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                    if (!moved) { int3 d = pos + new int3(0, -1, -dz); if (VoxelSimUtil.IsInBounds(d, worldSize) && ReadMatAt(d) == VoxelSimUtil.MAT_AIR) moved = TryClaim(d, voxel); }
                }

                // 3. Horizontal spread
                if (!moved)
                {
                    int spreadChance = (mat == VoxelSimUtil.MAT_WATER) ? 3 : 1;

                    if ((rng % 4u) < (uint)spreadChance)
                    {
                        int dx2 = ((rng >> 2) & 1u) != 0 ? 1 : -1;
                        int dz2 = ((rng >> 3) & 1u) != 0 ? 1 : -1;

                        if (!moved) { int3 h = pos + new int3(dx2, 0, 0);  if (VoxelSimUtil.IsInBounds(h, worldSize) && ReadMatAt(h) == VoxelSimUtil.MAT_AIR) moved = TryClaim(h, voxel); }
                        if (!moved) { int3 h = pos + new int3(0, 0, dz2);  if (VoxelSimUtil.IsInBounds(h, worldSize) && ReadMatAt(h) == VoxelSimUtil.MAT_AIR) moved = TryClaim(h, voxel); }
                        if (!moved) { int3 h = pos + new int3(-dx2, 0, 0); if (VoxelSimUtil.IsInBounds(h, worldSize) && ReadMatAt(h) == VoxelSimUtil.MAT_AIR) moved = TryClaim(h, voxel); }
                        if (!moved) { int3 h = pos + new int3(0, 0, -dz2); if (VoxelSimUtil.IsInBounds(h, worldSize) && ReadMatAt(h) == VoxelSimUtil.MAT_AIR) moved = TryClaim(h, voxel); }
                    }
                }

                // 4. Stay in place
                if (!moved) TryClaim(pos, voxel);
                return;
            }

            // --- Gas Physics (Steam) ---
            if (physCat == VoxelSimUtil.PHYS_GAS)
            {
                // Rise up
                int3 above = pos + new int3(0, 1, 0);
                if (VoxelSimUtil.IsInBounds(above, worldSize) &&
                    ReadMatAt(above) == VoxelSimUtil.MAT_AIR)
                {
                    moved = TryClaim(above, voxel);
                    if (moved) return;
                }

                // Diagonal up
                if (!moved)
                {
                    int dx = (rng & 1u) != 0 ? 1 : -1;
                    int3 diagUp = pos + new int3(dx, 1, 0);
                    if (VoxelSimUtil.IsInBounds(diagUp, worldSize) &&
                        ReadMatAt(diagUp) == VoxelSimUtil.MAT_AIR)
                    {
                        moved = TryClaim(diagUp, voxel);
                        if (moved) return;
                    }
                }

                // Dissipate: temperature acts as cooldown counter
                uint steamTemp = VoxelSimUtil.GetTemperature(voxel);
                if (steamTemp >= 15u)
                    return; // disappear (write buffer already 0)

                if (!moved)
                {
                    uint newVoxel = VoxelSimUtil.SetTemperature(voxel, steamTemp + 1u);
                    TryClaim(pos, newVoxel);
                }
                return;
            }

            // Fallback: claim current position
            TryClaim(pos, voxel);
        }

        // --- Private Helpers ---

        private uint ReadAt(int3 pos)
        {
            if (!VoxelSimUtil.IsInBounds(pos, worldSize)) return 0;
            return readBuffer[VoxelSimUtil.Flatten3D(pos, worldSize)];
        }

        private uint ReadMatAt(int3 pos)
        {
            return VoxelSimUtil.GetMaterialId(ReadAt(pos));
        }

        private bool HasNeighborMat(int3 pos, uint targetMat)
        {
            bool found = false;
            int3 n;
            n = pos + new int3( 1, 0, 0); found |= VoxelSimUtil.IsInBounds(n, worldSize) && ReadMatAt(n) == targetMat;
            n = pos + new int3(-1, 0, 0); found |= VoxelSimUtil.IsInBounds(n, worldSize) && ReadMatAt(n) == targetMat;
            n = pos + new int3( 0, 1, 0); found |= VoxelSimUtil.IsInBounds(n, worldSize) && ReadMatAt(n) == targetMat;
            n = pos + new int3( 0,-1, 0); found |= VoxelSimUtil.IsInBounds(n, worldSize) && ReadMatAt(n) == targetMat;
            n = pos + new int3( 0, 0, 1); found |= VoxelSimUtil.IsInBounds(n, worldSize) && ReadMatAt(n) == targetMat;
            n = pos + new int3( 0, 0,-1); found |= VoxelSimUtil.IsInBounds(n, worldSize) && ReadMatAt(n) == targetMat;
            return found;
        }

        /// <summary>
        /// Atomic CAS claim — mirrors GPU InterlockedCompareExchange.
        /// Attempts to write voxel into write buffer at pos. Returns true if
        /// the target was empty (0) and successfully claimed.
        /// </summary>
        private bool TryClaim(int3 pos, uint voxel)
        {
            if (!VoxelSimUtil.IsInBounds(pos, worldSize)) return false;
            int idx = VoxelSimUtil.Flatten3D(pos, worldSize);
            int original = Interlocked.CompareExchange(ref writeBufferPtr[idx], (int)voxel, 0);
            if (original == 0)
            {
                MarkBrickDirty(pos);
                return true;
            }
            return false;
        }

        private void MarkBrickDirty(int3 voxelPos)
        {
            int3 bp = voxelPos / brickSize;
            if (math.all(bp >= 0) & math.all(bp < brickMapSize))
            {
                int bi = VoxelSimUtil.Flatten3D(bp, brickMapSize);
                dirtyFlagsPtr[bi] = 1u; // Idempotent write — safe without atomics
            }
        }

        private uint Rand(int3 pos)
        {
            return VoxelSimUtil.HashPos(pos, frameCount * 7919u + simStep * 104729u);
        }
    }

    // =========================================================================
    // PropagateHeatAndLightJob — Heat diffusion + light propagation + state changes
    // Mirrors PropagateHeatAndLight kernel
    // No CAS needed — each thread writes only to its own index
    // =========================================================================

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast)]
    public struct PropagateHeatAndLightJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> readBuffer;
        public NativeArray<uint> writeBuffer;

        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<uint> dirtyFlags;

        public int worldSize;
        public int brickSize;
        public int brickMapSize;
        public uint frameCount;

        // Fire / heat tuning params
        public int fireSpreadNeighborTemp;
        public int woodCharTemp;
        public int leafBurnTemp;
        public int coalBurnoutTemp;
        public int snowMeltTemp;
        public int waterEvapTemp;
        public int burningLightTemp;
        public int heatRiseRate;
        public int coolRate;
        public int heatSinkExtraCool;
        public int coalBurnoutChanceDiv;

        public void Execute(int index)
        {
            int3 pos = VoxelSimUtil.Unflatten3D(index, worldSize);

            uint voxel = readBuffer[index];
            uint mat = VoxelSimUtil.GetMaterialId(voxel);

            if (mat == VoxelSimUtil.MAT_AIR)
            {
                writeBuffer[index] = 0u;
                return;
            }

            // --- Gather neighbor info ---
            uint maxNTemp = 0;
            uint maxNLight = 0;
            GatherNeighbor(pos + new int3( 1, 0, 0), ref maxNTemp, ref maxNLight);
            GatherNeighbor(pos + new int3(-1, 0, 0), ref maxNTemp, ref maxNLight);
            GatherNeighbor(pos + new int3( 0, 1, 0), ref maxNTemp, ref maxNLight);
            GatherNeighbor(pos + new int3( 0,-1, 0), ref maxNTemp, ref maxNLight);
            GatherNeighbor(pos + new int3( 0, 0, 1), ref maxNTemp, ref maxNLight);
            GatherNeighbor(pos + new int3( 0, 0,-1), ref maxNTemp, ref maxNLight);

            // --- Compute new temperature ---
            uint currentTemp = VoxelSimUtil.GetTemperature(voxel);
            uint newTemp = currentTemp;

            uint heatSource = VoxelSimUtil.GetMaterialHeatSource(mat);
            if (heatSource > 0u)
            {
                newTemp = heatSource;
            }
            else
            {
                uint incomingHeat = maxNTemp > 1u ? maxNTemp - 1u : 0u;

                if ((mat == VoxelSimUtil.MAT_WOOD || mat == VoxelSimUtil.MAT_LEAF || mat == VoxelSimUtil.MAT_COAL)
                    && maxNTemp >= (uint)fireSpreadNeighborTemp)
                    incomingHeat = math.max(incomingHeat, maxNTemp - 1u);

                if (incomingHeat > currentTemp)
                    newTemp = math.min(currentTemp + (uint)heatRiseRate, incomingHeat);
                else if (incomingHeat < currentTemp)
                    newTemp = currentTemp > (uint)coolRate ? currentTemp - (uint)coolRate : 0u;

                if (VoxelSimUtil.IsHeatSink(mat))
                {
                    if (newTemp > (uint)heatSinkExtraCool) newTemp -= (uint)heatSinkExtraCool;
                    else newTemp = 0u;
                    if (mat == VoxelSimUtil.MAT_WATER && newTemp > 10u) newTemp = 10u;
                    if (mat == VoxelSimUtil.MAT_SNOW  && newTemp >  3u) newTemp = 3u;
                }
            }

            newTemp = math.clamp(newTemp, 0u, 15u);

            // --- Heat-induced state changes ---
            if (mat == VoxelSimUtil.MAT_WOOD && newTemp >= (uint)woodCharTemp)
            {
                uint coalVoxel = VoxelSimUtil.PackVoxel(VoxelSimUtil.MAT_COAL, 0, 0);
                uint coalLight = math.max(newTemp > 4u ? newTemp - 4u : 0u, 3u);
                coalVoxel = VoxelSimUtil.SetTempAndLight(coalVoxel, math.max(newTemp, 9u), coalLight);
                writeBuffer[index] = coalVoxel;
                MarkBrickDirty(pos);
                return;
            }

            if (mat == VoxelSimUtil.MAT_LEAF && newTemp >= (uint)leafBurnTemp)
            {
                writeBuffer[index] = 0u;
                MarkBrickDirty(pos);
                return;
            }

            if (mat == VoxelSimUtil.MAT_COAL && newTemp >= (uint)coalBurnoutTemp)
            {
                uint chanceDiv = (uint)math.max(2, coalBurnoutChanceDiv);
                uint roll = VoxelSimUtil.HashPos(pos, frameCount * 1664525u + 1013904223u) % chanceDiv;
                if (roll == 0u)
                {
                    writeBuffer[index] = 0u;
                    MarkBrickDirty(pos);
                    return;
                }
            }

            if (mat == VoxelSimUtil.MAT_SNOW && newTemp >= (uint)snowMeltTemp)
            {
                uint waterVoxel = VoxelSimUtil.PackVoxel(VoxelSimUtil.MAT_WATER, 0, 0);
                waterVoxel = VoxelSimUtil.SetTempAndLight(waterVoxel, 0, 0);
                writeBuffer[index] = waterVoxel;
                MarkBrickDirty(pos);
                return;
            }

            if (mat == VoxelSimUtil.MAT_WATER && newTemp >= (uint)waterEvapTemp)
            {
                uint steamVoxel = VoxelSimUtil.PackVoxel(VoxelSimUtil.MAT_STEAM, 0, 0);
                steamVoxel = VoxelSimUtil.SetTempAndLight(steamVoxel, 0, 0);
                writeBuffer[index] = steamVoxel;
                MarkBrickDirty(pos);
                return;
            }

            // --- Compute new light level ---
            uint newLight = ComputeNewLight(mat, newTemp, maxNLight);

            voxel = VoxelSimUtil.SetTempAndLight(voxel, newTemp, newLight);
            writeBuffer[index] = voxel;
        }

        private uint ComputeNewLight(uint mat, uint temp, uint maxNeighborLight)
        {
            uint newLight = 0u;

            uint emission = VoxelSimUtil.GetMaterialEmission(mat);
            if (emission > 0u)
            {
                newLight = emission;
            }
            else if (mat == VoxelSimUtil.MAT_COAL && temp > 4u)
            {
                newLight = math.min(temp - 2u, 10u);
            }
            else if ((mat == VoxelSimUtil.MAT_WOOD || mat == VoxelSimUtil.MAT_LEAF)
                     && temp >= (uint)burningLightTemp)
            {
                newLight = math.min(temp - 1u, 11u);
            }
            else if (!VoxelSimUtil.BlocksLight(mat))
            {
                newLight = maxNeighborLight > 1u ? maxNeighborLight - 1u : 0u;
            }
            else
            {
                newLight = maxNeighborLight > 2u ? maxNeighborLight - 2u : 0u;
            }

            return math.clamp(newLight, 0u, 15u);
        }

        private void GatherNeighbor(int3 nPos, ref uint maxTemp, ref uint maxLight)
        {
            if (!VoxelSimUtil.IsInBounds(nPos, worldSize)) return;
            uint nVoxel = readBuffer[VoxelSimUtil.Flatten3D(nPos, worldSize)];
            uint nTemp = VoxelSimUtil.GetTemperature(nVoxel);
            uint nLight = VoxelSimUtil.GetLightLevel(nVoxel);
            if (nTemp > maxTemp) maxTemp = nTemp;
            if (nLight > maxLight) maxLight = nLight;
        }

        private void MarkBrickDirty(int3 voxelPos)
        {
            int3 bp = voxelPos / brickSize;
            if (math.all(bp >= 0) & math.all(bp < brickMapSize))
            {
                int bi = VoxelSimUtil.Flatten3D(bp, brickMapSize);
                dirtyFlags[bi] = 1u;
            }
        }
    }

    // =========================================================================
    // PropagateLightOnlyJob — Light convergence pass (no heat changes)
    // Mirrors PropagateLightOnly kernel
    // Optimized: early-exit for voxels in completely dark regions (~90% of world)
    // =========================================================================

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast)]
    public struct PropagateLightOnlyJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> readBuffer;
        public NativeArray<uint> writeBuffer;

        public int worldSize;
        public int burningLightTemp;

        public void Execute(int index)
        {
            uint voxel = readBuffer[index];
            uint mat = VoxelSimUtil.GetMaterialId(voxel);

            if (mat == VoxelSimUtil.MAT_AIR)
            {
                writeBuffer[index] = 0u;
                return;
            }

            // --- Fast check: can this voxel emit light? ---
            uint selfTemp = VoxelSimUtil.GetTemperature(voxel);
            uint emission = VoxelSimUtil.GetMaterialEmission(mat);
            uint blt = (uint)math.max(0, burningLightTemp);

            bool canEmit = emission > 0u
                || (mat == VoxelSimUtil.MAT_COAL && selfTemp > 4u)
                || ((mat == VoxelSimUtil.MAT_WOOD || mat == VoxelSimUtil.MAT_LEAF) && selfTemp >= blt);

            // If can't emit and has no current light, check neighbors
            // before doing the full computation. Most voxels in the world
            // are in completely dark regions — skip them fast.
            uint selfLight = VoxelSimUtil.GetLightLevel(voxel);

            if (!canEmit && selfLight == 0u)
            {
                // Inline neighbor light check — avoid Unflatten3D cost for ~90% of voxels
                int ws = worldSize;
                int wsSq = ws * ws;

                // Check all 6 neighbors for any light. Use direct index math
                // (cheaper than Unflatten3D + Flatten3D round-trip).
                bool anyLight = false;

                int nx;
                // +X
                nx = index + 1;
                if ((index % ws) < ws - 1)
                    anyLight |= ((readBuffer[nx] >> 28) & 0xFu) > 0u;
                // -X
                if (!anyLight && (index % ws) > 0)
                {
                    nx = index - 1;
                    anyLight |= ((readBuffer[nx] >> 28) & 0xFu) > 0u;
                }
                // +Y
                if (!anyLight && ((index / ws) % ws) < ws - 1)
                {
                    nx = index + ws;
                    anyLight |= ((readBuffer[nx] >> 28) & 0xFu) > 0u;
                }
                // -Y
                if (!anyLight && ((index / ws) % ws) > 0)
                {
                    nx = index - ws;
                    anyLight |= ((readBuffer[nx] >> 28) & 0xFu) > 0u;
                }
                // +Z
                if (!anyLight && (index / wsSq) < ws - 1)
                {
                    nx = index + wsSq;
                    anyLight |= ((readBuffer[nx] >> 28) & 0xFu) > 0u;
                }
                // -Z
                if (!anyLight && (index / wsSq) > 0)
                {
                    nx = index - wsSq;
                    anyLight |= ((readBuffer[nx] >> 28) & 0xFu) > 0u;
                }

                if (!anyLight)
                {
                    // Completely dark region — just copy voxel with light=0 (already is)
                    writeBuffer[index] = voxel;
                    return;
                }
            }

            // --- Full computation (only for emitting or lit/near-lit voxels) ---
            int3 pos = VoxelSimUtil.Unflatten3D(index, worldSize);

            // Gather max neighbor light
            uint maxNLight = 0u;
            GatherLight(pos + new int3( 1, 0, 0), ref maxNLight);
            GatherLight(pos + new int3(-1, 0, 0), ref maxNLight);
            GatherLight(pos + new int3( 0, 1, 0), ref maxNLight);
            GatherLight(pos + new int3( 0,-1, 0), ref maxNLight);
            GatherLight(pos + new int3( 0, 0, 1), ref maxNLight);
            GatherLight(pos + new int3( 0, 0,-1), ref maxNLight);

            uint newLight = 0u;

            if (emission > 0u)
            {
                newLight = emission;
            }
            else if (mat == VoxelSimUtil.MAT_COAL && selfTemp > 4u)
            {
                newLight = math.min(selfTemp - 2u, 10u);
            }
            else if ((mat == VoxelSimUtil.MAT_WOOD || mat == VoxelSimUtil.MAT_LEAF) && selfTemp >= blt)
            {
                newLight = math.min(selfTemp - 1u, 11u);
            }
            else if (!VoxelSimUtil.BlocksLight(mat))
            {
                newLight = maxNLight > 1u ? maxNLight - 1u : 0u;
            }
            else
            {
                newLight = maxNLight > 2u ? maxNLight - 2u : 0u;
            }

            newLight = math.clamp(newLight, 0u, 15u);

            voxel = VoxelSimUtil.SetLightLevel(voxel, newLight);
            writeBuffer[index] = voxel;
        }

        private void GatherLight(int3 nPos, ref uint maxLight)
        {
            if (!VoxelSimUtil.IsInBounds(nPos, worldSize)) return;
            uint nLight = VoxelSimUtil.GetLightLevel(readBuffer[VoxelSimUtil.Flatten3D(nPos, worldSize)]);
            if (nLight > maxLight) maxLight = nLight;
        }
    }

    // =========================================================================
    // UpdateBrickMapJob — Scan voxels per brick, write SVO level 0 occupancy
    // Mirrors UpdateBrickMap / UpdateBrickMapDirtyOnly kernels
    // =========================================================================

    [BurstCompile]
    public struct UpdateBrickMapJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> voxelBuffer;

        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<uint> svoBuffer;

        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<uint> dirtyFlags;

        public int worldSize;
        public int brickSize;
        public int brickMapSize;
        public int svoLevel0Offset;
        public bool fullRebuild;

        // Execute index = flat brick index
        public void Execute(int index)
        {
            if (!fullRebuild && dirtyFlags[index] == 0u)
                return;

            int3 brickPos = VoxelSimUtil.Unflatten3D(index, brickMapSize);
            int3 baseVoxel = brickPos * brickSize;

            bool hasContent = false;
            for (int z = 0; z < brickSize && !hasContent; z++)
            for (int y = 0; y < brickSize && !hasContent; y++)
            for (int x = 0; x < brickSize; x++)
            {
                int3 vPos = baseVoxel + new int3(x, y, z);
                if (VoxelSimUtil.IsInBounds(vPos, worldSize))
                {
                    uint voxel = voxelBuffer[VoxelSimUtil.Flatten3D(vPos, worldSize)];
                    if (VoxelSimUtil.GetMaterialId(voxel) != VoxelSimUtil.MAT_AIR)
                    {
                        hasContent = true;
                        break;
                    }
                }
            }

            svoBuffer[svoLevel0Offset + index] = hasContent ? 1u : 0u;

            // Clear dirty flag after processing
            if (!fullRebuild)
                dirtyFlags[index] = 0u;
        }
    }

    // =========================================================================
    // BuildSVOLevelJob — Reduce 2×2×2 children → 1 parent (OR)
    // Mirrors BuildSVOLevel kernel from SVOBuild.compute
    // =========================================================================

    [BurstCompile]
    public struct BuildSVOLevelJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<uint> svoBuffer;

        public int srcLevelOffset;
        public int dstLevelOffset;
        public int srcGridSize;
        public int dstGridSize;

        // Execute index = flat dst cell index
        public void Execute(int index)
        {
            int3 dstPos = VoxelSimUtil.Unflatten3D(index, dstGridSize);
            if (math.any(dstPos >= dstGridSize)) return;

            int3 srcBase = dstPos * 2;
            uint result = 0u;

            for (int z = 0; z < 2; z++)
            for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
            {
                int3 srcPos = srcBase + new int3(x, y, z);
                if (math.all(srcPos < srcGridSize))
                {
                    result |= svoBuffer[srcLevelOffset + VoxelSimUtil.Flatten3D(srcPos, srcGridSize)];
                }
            }

            svoBuffer[dstLevelOffset + VoxelSimUtil.Flatten3D(dstPos, dstGridSize)] = result;
        }
    }

    // =========================================================================
    // ClearDirtyFlagsJob / MarkAllDirtyJob — Utility jobs
    // =========================================================================

    [BurstCompile]
    public struct ClearDirtyFlagsJob : IJobParallelFor
    {
        public NativeArray<uint> dirtyFlags;
        public void Execute(int index) { dirtyFlags[index] = 0u; }
    }

    [BurstCompile]
    public struct MarkAllDirtyJob : IJobParallelFor
    {
        public NativeArray<uint> dirtyFlags;
        public void Execute(int index) { dirtyFlags[index] = 1u; }
    }

    // =========================================================================
    // FillPaddedVoxelDataJob — For VoxelCollision: fill padded region from
    // NativeArray world data using Burst parallelism
    // =========================================================================

    [BurstCompile]
    public struct FillPaddedVoxelDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> worldData;

        [NativeDisableParallelForRestriction]
        public NativeArray<uint> paddedData;

        public int paddedSizeX;
        public int paddedSizeY;
        public int minX;
        public int minY;
        public int minZ;
        public int worldSize;

        // Execute index = flat padded index
        public void Execute(int index)
        {
            int zStride = paddedSizeX * paddedSizeY;
            int pz = index / zStride;
            int rem = index - pz * zStride;
            int py = rem / paddedSizeX;
            int px = rem - py * paddedSizeX;

            int wx = minX + px - 1;
            int wy = minY + py - 1;
            int wz = minZ + pz - 1;

            if (wx < 0 || wx >= worldSize || wy < 0 || wy >= worldSize || wz < 0 || wz >= worldSize)
            {
                paddedData[index] = 0u;
                return;
            }

            int worldIdx = wx + wy * worldSize + wz * worldSize * worldSize;
            paddedData[index] = worldData[worldIdx];
        }
    }
}
