#ifndef VOXEL_COMMON_INCLUDED
#define VOXEL_COMMON_INCLUDED

// ============================================================================
// VoxelCommon.hlsl - Shared utilities for all voxel shaders
// Voxel format: 32-bit packed
//   Bits  0-7 : Material ID (256 types)
//   Bits  8-23: Color RGB565 (16 bits)
//   Bits 24-31: Auxiliary data (8 bits)
// ============================================================================

// --- Material ID Constants ---
#define MAT_AIR    0
#define MAT_STONE  1
#define MAT_DIRT   2
#define MAT_GRASS  3
#define MAT_SAND   4
#define MAT_WATER  5
#define MAT_LAVA   6
#define MAT_WOOD   7
#define MAT_LEAF   8
#define MAT_SNOW   9
#define MAT_GLASS  10
#define MAT_IRON   11
#define MAT_COAL   12
#define MAT_GOLD   13
#define MAT_STEAM  14
#define MAT_COUNT  15

// --- Material Physics Categories ---
#define PHYS_STATIC   0  // Stone, Dirt, Grass, Wood, etc.
#define PHYS_POWDER   1  // Sand, Snow, Coal
#define PHYS_LIQUID   2  // Water, Lava
#define PHYS_GAS      3  // Steam

// --- Packing / Unpacking ---

uint PackVoxel(uint materialId, uint color565, uint aux)
{
    return (aux << 24) | (color565 << 8) | (materialId & 0xFF);
}

uint GetMaterialId(uint voxel)
{
    return voxel & 0xFF;
}

uint GetColor565(uint voxel)
{
    return (voxel >> 8) & 0xFFFF;
}

uint GetAuxData(uint voxel)
{
    return (voxel >> 24) & 0xFF;
}

uint SetMaterialId(uint voxel, uint matId)
{
    return (voxel & 0xFFFFFF00) | (matId & 0xFF);
}

uint SetColor565(uint voxel, uint color)
{
    return (voxel & 0xFF0000FF) | ((color & 0xFFFF) << 8);
}

uint SetAuxData(uint voxel, uint aux)
{
    return (voxel & 0x00FFFFFF) | ((aux & 0xFF) << 24);
}

// --- Color Conversion ---

float3 Color565ToRGB(uint c)
{
    float r = float((c >> 11) & 0x1F) / 31.0;
    float g = float((c >> 5) & 0x3F)  / 63.0;
    float b = float(c & 0x1F)         / 31.0;
    return float3(r, g, b);
}

uint RGBToColor565(float3 color)
{
    uint r = (uint)(saturate(color.r) * 31.0 + 0.5);
    uint g = (uint)(saturate(color.g) * 63.0 + 0.5);
    uint b = (uint)(saturate(color.b) * 31.0 + 0.5);
    return (r << 11) | (g << 5) | b;
}

// --- 3D Index Helpers ---

int Flatten3D(int3 pos, int size)
{
    return pos.x + pos.y * size + pos.z * size * size;
}

int3 Unflatten3D(int index, int size)
{
    int z = index / (size * size);
    int rem = index % (size * size);
    int y = rem / size;
    int x = rem % size;
    return int3(x, y, z);
}

bool IsInBounds(int3 pos, int size)
{
    return all(pos >= 0) && all(pos < size);
}

// --- Default Material Colors ---

float3 GetDefaultMaterialColor(uint matId)
{
    switch (matId)
    {
        case MAT_STONE: return float3(0.50, 0.50, 0.52);
        case MAT_DIRT:  return float3(0.55, 0.36, 0.22);
        case MAT_GRASS: return float3(0.30, 0.60, 0.20);
        case MAT_SAND:  return float3(0.90, 0.85, 0.60);
        case MAT_WATER: return float3(0.20, 0.40, 0.80);
        case MAT_LAVA:  return float3(1.00, 0.35, 0.05);
        case MAT_WOOD:  return float3(0.55, 0.36, 0.18);
        case MAT_LEAF:  return float3(0.22, 0.52, 0.14);
        case MAT_SNOW:  return float3(0.94, 0.94, 0.97);
        case MAT_GLASS: return float3(0.75, 0.85, 0.90);
        case MAT_IRON:  return float3(0.60, 0.58, 0.56);
        case MAT_COAL:  return float3(0.20, 0.20, 0.20);
        case MAT_GOLD:  return float3(0.95, 0.82, 0.20);
        case MAT_STEAM: return float3(0.85, 0.85, 0.90);
        default:        return float3(1.0, 0.0, 1.0);
    }
}

// --- Material Physics Category ---

uint GetPhysicsCategory(uint matId)
{
    if (matId == MAT_SAND || matId == MAT_SNOW || matId == MAT_COAL)
        return PHYS_POWDER;
    if (matId == MAT_WATER || matId == MAT_LAVA)
        return PHYS_LIQUID;
    if (matId == MAT_STEAM)
        return PHYS_GAS;
    return PHYS_STATIC;
}

bool IsTransparent(uint matId)
{
    return matId == MAT_AIR || matId == MAT_WATER || matId == MAT_GLASS || matId == MAT_STEAM;
}

bool IsSolid(uint matId)
{
    return matId != MAT_AIR && matId != MAT_WATER && matId != MAT_LAVA && matId != MAT_STEAM;
}

// --- Hashing (for pseudo-random in shaders) ---

uint VoxelHash(uint x)
{
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

uint HashPos(int3 pos, uint seed)
{
    uint h = (uint)pos.x * 73856093u;
    h ^= (uint)pos.y * 19349663u;
    h ^= (uint)pos.z * 83492791u;
    h ^= seed * 1299709u;
    return VoxelHash(h);
}

float HashToFloat(uint h)
{
    return float(h & 0xFFFF) / 65535.0;
}

#endif // VOXEL_COMMON_INCLUDED
