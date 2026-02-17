#ifndef VOXEL_NOISE_INCLUDED
#define VOXEL_NOISE_INCLUDED

// ============================================================================
// VoxelNoise.hlsl - Noise functions for terrain generation
// Hash-based value noise with FBM (Fractal Brownian Motion)
// ============================================================================

// --- Hash-based 3D noise ---

float _NoiseHash3D(float3 p)
{
    p = frac(p * float3(443.8975, 441.4231, 437.1953));
    p += dot(p, p.yzx + 19.19);
    return frac((p.x + p.y) * p.z);
}

float ValueNoise3D(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);
    
    // Smooth interpolation (Hermite)
    f = f * f * (3.0 - 2.0 * f);
    
    // 8 corner samples
    float c000 = _NoiseHash3D(i + float3(0, 0, 0));
    float c100 = _NoiseHash3D(i + float3(1, 0, 0));
    float c010 = _NoiseHash3D(i + float3(0, 1, 0));
    float c110 = _NoiseHash3D(i + float3(1, 1, 0));
    float c001 = _NoiseHash3D(i + float3(0, 0, 1));
    float c101 = _NoiseHash3D(i + float3(1, 0, 1));
    float c011 = _NoiseHash3D(i + float3(0, 1, 1));
    float c111 = _NoiseHash3D(i + float3(1, 1, 1));
    
    // Trilinear interpolation
    return lerp(
        lerp(lerp(c000, c100, f.x), lerp(c010, c110, f.x), f.y),
        lerp(lerp(c001, c101, f.x), lerp(c011, c111, f.x), f.y),
        f.z
    );
}

// --- Fractal Brownian Motion ---

float FBM3D(float3 p, int octaves, float lacunarity, float persistence)
{
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++)
    {
        value += amplitude * ValueNoise3D(p * frequency);
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / maxValue; // Normalize to [0, 1]
}

// --- 2D noise (for heightmaps) ---

float _NoiseHash2D(float2 p)
{
    p = frac(p * float2(443.8975, 441.4231));
    p += dot(p, p.yx + 19.19);
    return frac(p.x * p.y);
}

float ValueNoise2D(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float c00 = _NoiseHash2D(i + float2(0, 0));
    float c10 = _NoiseHash2D(i + float2(1, 0));
    float c01 = _NoiseHash2D(i + float2(0, 1));
    float c11 = _NoiseHash2D(i + float2(1, 1));
    
    return lerp(
        lerp(c00, c10, f.x),
        lerp(c01, c11, f.x),
        f.y
    );
}

float FBM2D(float2 p, int octaves, float lacunarity, float persistence)
{
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++)
    {
        value += amplitude * ValueNoise2D(p * frequency);
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / maxValue;
}

// --- Ridged Noise (for mountains / caves) ---

float RidgedNoise3D(float3 p, int octaves, float lacunarity, float persistence)
{
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;
    
    for (int i = 0; i < octaves; i++)
    {
        float n = 1.0 - abs(ValueNoise3D(p * frequency) * 2.0 - 1.0);
        value += amplitude * n * n;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / maxValue;
}

#endif // VOXEL_NOISE_INCLUDED
