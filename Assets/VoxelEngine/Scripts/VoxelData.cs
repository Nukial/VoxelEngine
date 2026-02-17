using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Static utility class for voxel data packing/unpacking.
    /// Voxel format: 32-bit packed uint
    ///   Bits  0-7 : Material ID (256 types)
    ///   Bits  8-23: Color RGB565 (16 bits)  
    ///   Bits 24-31: Auxiliary data (8 bits)
    /// </summary>
    public static class VoxelData
    {
        // Material IDs - must match VoxelCommon.hlsl
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
        public const uint MAT_COUNT = 15;

        // Material names for UI
        public static readonly string[] MaterialNames = {
            "Air", "Stone", "Dirt", "Grass", "Sand",
            "Water", "Lava", "Wood", "Leaf", "Snow",
            "Glass", "Iron", "Coal", "Gold", "Steam"
        };

        // Default colors for each material
        public static readonly Color[] MaterialColors = {
            new Color(0f, 0f, 0f, 0f),         // Air
            new Color(0.50f, 0.50f, 0.52f),     // Stone
            new Color(0.55f, 0.36f, 0.22f),     // Dirt
            new Color(0.30f, 0.60f, 0.20f),     // Grass
            new Color(0.90f, 0.85f, 0.60f),     // Sand
            new Color(0.20f, 0.40f, 0.80f),     // Water
            new Color(1.00f, 0.35f, 0.05f),     // Lava
            new Color(0.55f, 0.36f, 0.18f),     // Wood
            new Color(0.22f, 0.52f, 0.14f),     // Leaf
            new Color(0.94f, 0.94f, 0.97f),     // Snow
            new Color(0.75f, 0.85f, 0.90f),     // Glass
            new Color(0.60f, 0.58f, 0.56f),     // Iron
            new Color(0.20f, 0.20f, 0.20f),     // Coal
            new Color(0.95f, 0.82f, 0.20f),     // Gold
            new Color(0.85f, 0.85f, 0.90f),     // Steam
        };

        // --- Packing ---

        public static uint Pack(uint materialId, ushort color565, byte aux)
        {
            return ((uint)aux << 24) | ((uint)color565 << 8) | (materialId & 0xFF);
        }

        public static uint Pack(uint materialId, Color color, byte aux = 0)
        {
            return Pack(materialId, ColorToRGB565(color), aux);
        }

        public static uint PackWithDefaultColor(uint materialId, byte aux = 0)
        {
            Color c = materialId < MAT_COUNT ? MaterialColors[materialId] : Color.magenta;
            return Pack(materialId, c, aux);
        }

        // --- Unpacking ---

        public static uint GetMaterialId(uint voxel)
        {
            return voxel & 0xFF;
        }

        public static ushort GetColor565(uint voxel)
        {
            return (ushort)((voxel >> 8) & 0xFFFF);
        }

        public static byte GetAuxData(uint voxel)
        {
            return (byte)((voxel >> 24) & 0xFF);
        }

        // --- Color Conversion ---

        public static ushort ColorToRGB565(Color color)
        {
            uint r = (uint)(Mathf.Clamp01(color.r) * 31f + 0.5f);
            uint g = (uint)(Mathf.Clamp01(color.g) * 63f + 0.5f);
            uint b = (uint)(Mathf.Clamp01(color.b) * 31f + 0.5f);
            return (ushort)((r << 11) | (g << 5) | b);
        }

        public static Color RGB565ToColor(ushort color565)
        {
            float r = ((color565 >> 11) & 0x1F) / 31f;
            float g = ((color565 >> 5) & 0x3F) / 63f;
            float b = (color565 & 0x1F) / 31f;
            return new Color(r, g, b, 1f);
        }

        // --- 3D Indexing ---

        public static int Flatten3D(int x, int y, int z, int size)
        {
            return x + y * size + z * size * size;
        }

        public static int Flatten3D(Vector3Int pos, int size)
        {
            return pos.x + pos.y * size + pos.z * size * size;
        }

        public static Vector3Int Unflatten3D(int index, int size)
        {
            int z = index / (size * size);
            int rem = index % (size * size);
            int y = rem / size;
            int x = rem % size;
            return new Vector3Int(x, y, z);
        }

        public static bool IsInBounds(Vector3Int pos, int size)
        {
            return pos.x >= 0 && pos.x < size &&
                   pos.y >= 0 && pos.y < size &&
                   pos.z >= 0 && pos.z < size;
        }

        // --- Material Queries ---

        public static bool IsStatic(uint materialId)
        {
            return materialId == MAT_STONE || materialId == MAT_DIRT ||
                   materialId == MAT_GRASS || materialId == MAT_WOOD ||
                   materialId == MAT_LEAF || materialId == MAT_IRON ||
                   materialId == MAT_GOLD || materialId == MAT_GLASS;
        }

        public static bool IsSolid(uint materialId)
        {
            return materialId != MAT_AIR && materialId != MAT_WATER &&
                   materialId != MAT_LAVA && materialId != MAT_STEAM;
        }

        public static bool IsLiquid(uint materialId)
        {
            return materialId == MAT_WATER || materialId == MAT_LAVA;
        }

        public static bool IsPowder(uint materialId)
        {
            return materialId == MAT_SAND || materialId == MAT_SNOW || materialId == MAT_COAL;
        }
    }
}
