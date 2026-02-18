Shader "VoxelEngine/RayMarch"
{
    Properties
    {
        [Header(Lighting)]
        _AmbientColor ("Ambient Color", Color) = (0.15, 0.18, 0.25, 1)
        _SunColor ("Sun Color", Color) = (1.0, 0.95, 0.85, 1)
        _SunIntensity ("Sun Intensity", Float) = 1.2
        _SpecularPower ("Specular Power", Float) = 32
        
        [Header(Rendering)]
        _MaxSteps ("Max Ray Steps", Int) = 512
        _MaxShadowSteps ("Max Shadow Steps", Int) = 128
        _FogDensity ("Fog Density", Float) = 0.003
        _FogColor ("Fog Color", Color) = (0.6, 0.75, 0.9, 1)
        _MaxRenderDist ("Max Render Distance", Float) = 80
        _ShadowStrength ("Shadow Strength", Range(0, 1)) = 1.0
        
        [Header(Culling)]
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull Mode", Float) = 1
    }
    
    SubShader
    {
        Tags 
        { 
            "RenderType" = "Opaque" 
            "RenderPipeline" = "UniversalPipeline" 
            "Queue" = "Geometry+10" 
        }
        
        Pass
        {
            Name "VoxelRayMarch"
            Tags { "LightMode" = "UniversalForward" }
            
            Cull [_Cull]
            ZWrite On
            ZTest LEqual
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ VOXEL_SHADOWS_ON
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "VoxelCommon.hlsl"
            
            // --- Buffers ---
            StructuredBuffer<uint> _VoxelBuffer;
            StructuredBuffer<uint> _SVOBuffer;
            
            // --- Parameters (set from C#) ---
            int _WorldSize;
            int _BrickSize;
            int _BrickMapSize;
            float _VoxelScale;
            float3 _WorldOrigin;   // World-space origin of the voxel volume
            int _MaxSteps;
            int _MaxShadowSteps;
            
            // SVO hierarchy parameters
            int4 _SVOLevelOffsets;   // offsets for levels 0-3
            int2 _SVOLevelOffsets2;  // offsets for levels 4-5
            int _SVOLevelCount;      // number of valid levels
            
            // Lighting
            float4 _AmbientColor;
            float4 _SunColor;
            float _SunIntensity;
            float _SpecularPower;
            float3 _SunDir;
            
            // Fog
            float _FogDensity;
            float4 _FogColor;
            
            // View distance
            float _MaxRenderDist;
            
            // Shadow blending
            float _ShadowStrength;
            
            // --- Structures ---
            
            struct Attributes
            {
                float4 positionOS : POSITION;
            };
            
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
            };
            
            struct FragOutput
            {
                float4 color : SV_Target;
                float depth : SV_Depth;
            };
            
            struct RayHit
            {
                bool hit;
                int3 voxelPos;
                float3 normal;
                float t;
                uint voxelData;
            };
            
            // Transparent layer for front-to-back blending
            struct TransparentLayer
            {
                float3 color;
                float opacity;
                int3 voxelPos;
                float3 normal;
                uint voxelData;
            };
            
            #define MAX_TRANSPARENT_LAYERS 6
            
            // --- Vertex Shader ---
            
            Varyings vert(Attributes input)
            {
                Varyings o;
                o.positionWS = TransformObjectToWorld(input.positionOS.xyz);
                o.positionCS = TransformWorldToHClip(o.positionWS);
                return o;
            }
            
            // --- Ray-AABB Intersection ---
            
            bool IntersectAABB(float3 ro, float3 rd, float3 bmin, float3 bmax,
                               out float tNear, out float tFar)
            {
                float3 invD = 1.0 / rd;
                float3 t0 = (bmin - ro) * invD;
                float3 t1 = (bmax - ro) * invD;
                float3 tmin = min(t0, t1);
                float3 tmax = max(t0, t1);
                tNear = max(max(tmin.x, tmin.y), tmin.z);
                tFar  = min(min(tmax.x, tmax.y), tmax.z);
                return tNear <= tFar && tFar > 0.0;
            }
            
            // --- Voxel Access ---
            
            uint ReadVoxel(int3 pos)
            {
                if (any(pos < 0) || any(pos >= _WorldSize))
                    return 0;
                return _VoxelBuffer[Flatten3D(pos, _WorldSize)];
            }
            
            // --- SVO Hierarchical Empty-Space Query ---
            // Returns the level offset for a given SVO level (0-5)
            int GetSVOLevelOffset(int level)
            {
                if (level < 4) 
                {
                    // _SVOLevelOffsets.xyzw for levels 0-3
                    if (level == 0) return _SVOLevelOffsets.x;
                    if (level == 1) return _SVOLevelOffsets.y;
                    if (level == 2) return _SVOLevelOffsets.z;
                    return _SVOLevelOffsets.w;
                }
                if (level == 4) return _SVOLevelOffsets2.x;
                return _SVOLevelOffsets2.y;
            }
            
            // Attempt hierarchical empty-space skip from coarsest to finest SVO level.
            // Returns the voxel-space cell size to skip, or 0 if no skip is possible.
            int TrySVOSkip(int3 pos)
            {
                // Check from coarsest level down to level 0 (brick level)
                // Stop at the first empty level found (biggest skip possible)
                int bestCellSize = 0;
                
                [unroll]
                for (int level = MAX_SVO_LEVELS - 1; level >= 0; level--)
                {
                    if (level >= _SVOLevelCount) continue;
                    if (bestCellSize > 0) continue; // already found a skip
                    
                    int cellVoxelSize = _BrickSize << level;
                    int gridSize = _WorldSize / cellVoxelSize;
                    int3 cellPos = pos / cellVoxelSize;
                    
                    if (any(cellPos < 0) || any(cellPos >= gridSize)) continue;
                    
                    int levelOffset = GetSVOLevelOffset(level);
                    if (_SVOBuffer[levelOffset + Flatten3D(cellPos, gridSize)] == 0)
                    {
                        bestCellSize = cellVoxelSize;
                    }
                }
                
                return bestCellSize;
            }
            
            // --- World <-> Voxel Space Conversion ---
            
            float3 WorldToVoxel(float3 worldPos)
            {
                return (worldPos - _WorldOrigin) / _VoxelScale;
            }
            
            float3 VoxelToWorld(float3 voxelPos)
            {
                return voxelPos * _VoxelScale + _WorldOrigin;
            }
            
            // --- Amanatides and Woo Ray Marching with Brick Map ---
            // Extended to support transparency: collects transparent layers
            // and continues marching until hitting an opaque surface.
            
            RayHit RayMarchVoxels(float3 origin, float3 dir, int maxSteps,
                                  out TransparentLayer transLayers[MAX_TRANSPARENT_LAYERS],
                                  out int transCount)
            {
                RayHit result;
                result.hit = false;
                result.t = 1e10;
                result.normal = float3(0, 1, 0);
                result.voxelPos = int3(0, 0, 0);
                result.voxelData = 0;
                
                transCount = 0;
                [unroll]
                for (int tl = 0; tl < MAX_TRANSPARENT_LAYERS; tl++)
                {
                    transLayers[tl].color = float3(0, 0, 0);
                    transLayers[tl].opacity = 0;
                    transLayers[tl].voxelPos = int3(0, 0, 0);
                    transLayers[tl].normal = float3(0, 1, 0);
                    transLayers[tl].voxelData = 0;
                }
                
                // AABB intersection with voxel volume [0, WorldSize]
                float tNear, tFar;
                float3 bmin = float3(0, 0, 0);
                float3 bmax = float3(_WorldSize, _WorldSize, _WorldSize);
                
                if (!IntersectAABB(origin, dir, bmin, bmax, tNear, tFar))
                    return result;
                
                tNear = max(tNear, 0.0001);
                
                // Max traversal distance in voxel space
                float maxVoxelDist = _MaxRenderDist / max(_VoxelScale, 0.001);
                // Clamp tFar to not exceed render distance from entry
                tFar = min(tFar, tNear + maxVoxelDist);
                
                // Entry point
                float3 startPos = origin + dir * tNear;
                startPos = clamp(startPos, 0.001, (float)_WorldSize - 0.001);
                
                // Current voxel position
                int3 pos = int3(floor(startPos));
                pos = clamp(pos, int3(0,0,0), int3(_WorldSize-1, _WorldSize-1, _WorldSize-1));
                
                // Step direction (set from safeDir below to avoid zero-step axes)
                int3 stepDir = int3(0, 0, 0);
                
                // Prevent division by zero
                float3 safeDir = dir;
                safeDir.x = abs(safeDir.x) < 1e-8 ? (safeDir.x >= 0 ? 1e-8 : -1e-8) : safeDir.x;
                safeDir.y = abs(safeDir.y) < 1e-8 ? (safeDir.y >= 0 ? 1e-8 : -1e-8) : safeDir.y;
                safeDir.z = abs(safeDir.z) < 1e-8 ? (safeDir.z >= 0 ? 1e-8 : -1e-8) : safeDir.z;

                stepDir = int3(
                    safeDir.x >= 0 ? 1 : -1,
                    safeDir.y >= 0 ? 1 : -1,
                    safeDir.z >= 0 ? 1 : -1
                );
                
                float3 invDir = 1.0 / safeDir;
                
                // tDelta: parametric distance to traverse one voxel
                float3 tDelta = abs(invDir);
                
                // tMax: parametric distance to next voxel boundary
                float3 nextBound;
                nextBound.x = stepDir.x > 0 ? float(pos.x + 1) : float(pos.x);
                nextBound.y = stepDir.y > 0 ? float(pos.y + 1) : float(pos.y);
                nextBound.z = stepDir.z > 0 ? float(pos.z + 1) : float(pos.z);
                
                float3 tMax = (nextBound - origin) * invDir;
                
                float3 normal = float3(0, 0, 0);
                
                for (int i = 0; i < maxSteps; i++)
                {
                    // Bounds check
                    if (any(pos < 0) || any(pos >= _WorldSize))
                        break;
                    
                    // Early termination: stop if ray has traveled beyond max render distance
                    float currentT = min(min(tMax.x, tMax.y), tMax.z);
                    if (currentT > tNear + maxVoxelDist)
                        break;
                    
                    // --- SVO Hierarchical Empty-Space Skip ---
                    // Try from coarsest to finest level; skip at the biggest empty cell
                    int skipCellSize = TrySVOSkip(pos);
                    
                    if (skipCellSize > 0)
                    {
                        // Compute the cell bounds and jump the ray past it
                        int3 cellPos = pos / skipCellSize;
                        float3 cellMinF = float3(cellPos * skipCellSize);
                        float3 cellMaxF = cellMinF + float(skipCellSize);
                        
                        float3 tExit0 = (cellMinF - origin) * invDir;
                        float3 tExit1 = (cellMaxF - origin) * invDir;
                        float3 tFarAxis = max(tExit0, tExit1);
                        float tCellExit = min(min(tFarAxis.x, tFarAxis.y), tFarAxis.z);
                        
                        if (tCellExit == tFarAxis.x) normal = float3(-stepDir.x, 0, 0);
                        else if (tCellExit == tFarAxis.y) normal = float3(0, -stepDir.y, 0);
                        else normal = float3(0, 0, -stepDir.z);
                        
                        float3 jumpPos = origin + safeDir * (tCellExit + 0.001);
                        pos = int3(floor(jumpPos));
                        
                        nextBound.x = stepDir.x > 0 ? float(pos.x + 1) : float(pos.x);
                        nextBound.y = stepDir.y > 0 ? float(pos.y + 1) : float(pos.y);
                        nextBound.z = stepDir.z > 0 ? float(pos.z + 1) : float(pos.z);
                        tMax = (nextBound - origin) * invDir;
                        
                        continue;
                    }
                    
                    // Sample voxel
                    uint voxel = ReadVoxel(pos);
                    uint matId = GetMaterialId(voxel);
                    
                    if (matId != MAT_AIR)
                    {
                        float matOpacity = GetMaterialOpacity(matId);
                        
                        // Transparent material: record layer and continue marching
                        if (matOpacity < 1.0 && transCount < MAX_TRANSPARENT_LAYERS)
                        {
                            // Get color for transparent layer
                            uint tc565 = GetColor565(voxel);
                            float3 tColor = (tc565 != 0) ? Color565ToRGB(tc565) : GetDefaultMaterialColor(matId);
                            
                            transLayers[transCount].color = tColor;
                            transLayers[transCount].opacity = matOpacity;
                            transLayers[transCount].voxelPos = pos;
                            transLayers[transCount].normal = normal;
                            transLayers[transCount].voxelData = voxel;
                            transCount++;
                            
                            // If first transparent hit, record t for depth
                            if (!result.hit)
                            {
                                float3 tLast = tMax - tDelta;
                                float firstT = max(max(tLast.x * (abs(normal.x) > 0.5 ? 1 : 0) + 
                                                       tLast.y * (abs(normal.y) > 0.5 ? 1 : 0) + 
                                                       tLast.z * (abs(normal.z) > 0.5 ? 1 : 0), 0), tNear);
                                result.t = firstT;
                                result.voxelPos = pos;
                                result.normal = normal;
                                result.voxelData = voxel;
                            }
                            
                            // Continue marching through transparent voxel
                        }
                        else
                        {
                            // Opaque hit
                            result.hit = true;
                            result.voxelPos = pos;
                            result.normal = normal;
                            result.voxelData = voxel;
                            
                            // Compute hit distance
                            float3 tLast = tMax - tDelta;
                            float hitT = max(max(tLast.x * (abs(normal.x) > 0.5 ? 1 : 0) + 
                                               tLast.y * (abs(normal.y) > 0.5 ? 1 : 0) + 
                                               tLast.z * (abs(normal.z) > 0.5 ? 1 : 0), 0), tNear);
                            
                            // If we had transparent layers before, keep the first t
                            if (transCount == 0)
                                result.t = hitT;
                            
                            // Fallback distance computation
                            if (length(normal) < 0.5)
                            {
                                if (transCount == 0)
                                    result.t = tNear;
                                result.normal = float3(0, 1, 0);
                            }
                            
                            return result;
                        }
                    }
                    
                    // Amanatides and Woo: step to next voxel
                    if (tMax.x < tMax.y)
                    {
                        if (tMax.x < tMax.z)
                        {
                            pos.x += stepDir.x;
                            tMax.x += tDelta.x;
                            normal = float3(-stepDir.x, 0, 0);
                        }
                        else
                        {
                            pos.z += stepDir.z;
                            tMax.z += tDelta.z;
                            normal = float3(0, 0, -stepDir.z);
                        }
                    }
                    else
                    {
                        if (tMax.y < tMax.z)
                        {
                            pos.y += stepDir.y;
                            tMax.y += tDelta.y;
                            normal = float3(0, -stepDir.y, 0);
                        }
                        else
                        {
                            pos.z += stepDir.z;
                            tMax.z += tDelta.z;
                            normal = float3(0, 0, -stepDir.z);
                        }
                    }
                }
                
                // If we only hit transparent layers with no opaque behind, still report a hit
                if (!result.hit && transCount > 0)
                {
                    result.hit = true;
                    // result.t was already set at first transparent hit
                }
                
                return result;
            }
            
            // --- Shadow Ray (simplified, no brick skip for perf) ---
            
            bool CastShadowRay(float3 origin, float3 dir, int maxSteps)
            {
                int3 pos = int3(floor(origin));
                int3 stepDir = int3(0, 0, 0);
                
                float3 safeDir = dir;
                safeDir.x = abs(safeDir.x) < 1e-8 ? 1e-8 : safeDir.x;
                safeDir.y = abs(safeDir.y) < 1e-8 ? 1e-8 : safeDir.y;
                safeDir.z = abs(safeDir.z) < 1e-8 ? 1e-8 : safeDir.z;

                stepDir = int3(
                    safeDir.x >= 0 ? 1 : -1,
                    safeDir.y >= 0 ? 1 : -1,
                    safeDir.z >= 0 ? 1 : -1
                );
                
                float3 invDir = 1.0 / safeDir;
                float3 tDelta = abs(invDir);
                
                // Limit shadow rays to a shorter distance for performance
                float maxShadowVoxelDist = min(_MaxRenderDist / max(_VoxelScale, 0.001) * 0.5, 64.0);
                
                float3 nextBound;
                nextBound.x = stepDir.x > 0 ? float(pos.x + 1) : float(pos.x);
                nextBound.y = stepDir.y > 0 ? float(pos.y + 1) : float(pos.y);
                nextBound.z = stepDir.z > 0 ? float(pos.z + 1) : float(pos.z);
                float3 tMax = (nextBound - origin) * invDir;
                
                for (int i = 0; i < maxSteps; i++)
                {
                    if (any(pos < 0) || any(pos >= _WorldSize))
                        return false; // Escaped to sky
                    
                    // Early termination for shadow distance
                    float shadowT = min(min(tMax.x, tMax.y), tMax.z);
                    if (shadowT > maxShadowVoxelDist)
                        return false;
                    
                    // SVO hierarchical empty-space skip for shadow rays
                    int skipCellSize = TrySVOSkip(pos);
                    
                    if (skipCellSize > 0)
                    {
                        int3 cellPos = pos / skipCellSize;
                        float3 cellMinF = float3(cellPos * skipCellSize);
                        float3 cellMaxF = cellMinF + float(skipCellSize);
                        
                        float3 tE0 = (cellMinF - origin) * invDir;
                        float3 tE1 = (cellMaxF - origin) * invDir;
                        float3 tFA = max(tE0, tE1);
                        float tCellExit = min(min(tFA.x, tFA.y), tFA.z);
                        
                        float3 jumpPos = origin + safeDir * (tCellExit + 0.001);
                        pos = int3(floor(jumpPos));
                        
                        nextBound.x = stepDir.x > 0 ? float(pos.x + 1) : float(pos.x);
                        nextBound.y = stepDir.y > 0 ? float(pos.y + 1) : float(pos.y);
                        nextBound.z = stepDir.z > 0 ? float(pos.z + 1) : float(pos.z);
                        tMax = (nextBound - origin) * invDir;
                        continue;
                    }
                    
                    uint voxel = ReadVoxel(pos);
                    if (GetMaterialId(voxel) != MAT_AIR && !IsTransparent(GetMaterialId(voxel)))
                        return true; // Hit an opaque voxel = in shadow
                    
                    if (tMax.x < tMax.y)
                    {
                        if (tMax.x < tMax.z) { pos.x += stepDir.x; tMax.x += tDelta.x; }
                        else                  { pos.z += stepDir.z; tMax.z += tDelta.z; }
                    }
                    else
                    {
                        if (tMax.y < tMax.z)  { pos.y += stepDir.y; tMax.y += tDelta.y; }
                        else                   { pos.z += stepDir.z; tMax.z += tDelta.z; }
                    }
                }
                
                return false;
            }
            
            // --- Shade a single voxel surface ---
            
            float3 ShadeVoxel(int3 voxelPos, float3 normal, uint voxelData,
                              float3 voxelDir, float3 L)
            {
                uint matId = GetMaterialId(voxelData);
                uint color565 = GetColor565(voxelData);
                
                // Base color
                float3 baseColor;
                if (color565 != 0)
                    baseColor = Color565ToRGB(color565);
                else
                    baseColor = GetDefaultMaterialColor(matId);
                
                // Subtle variation
                float variation = HashToFloat(HashPos(voxelPos, 42u));
                baseColor *= lerp(0.9, 1.1, variation);
                
                float3 N = normalize(normal);
                float3 V = normalize(-voxelDir);
                float3 H = normalize(L + V);
                
                // Diffuse
                float NdotL = max(dot(N, L), 0.0);
                
                // Specular
                float NdotH = max(dot(N, H), 0.0);
                float specIntensity = (matId == MAT_WATER || matId == MAT_GLASS) ? 0.6 :
                                      (matId == MAT_LAVA) ? 0.0 :
                                      (matId == MAT_IRON || matId == MAT_GOLD) ? 0.4 : 0.08;
                float spec = pow(NdotH, _SpecularPower) * specIntensity;
                
                // Shadow
                float shadowFactor = 1.0;
                #ifdef VOXEL_SHADOWS_ON
                {
                    float3 shadowOrigin = float3(voxelPos) + 0.5 + normal * 1.2 + L * 0.15;
                    if (CastShadowRay(shadowOrigin, L, _MaxShadowSteps))
                    {
                        float shadowDarkness = lerp(1.0, 0.25, _ShadowStrength);
                        shadowFactor = shadowDarkness;
                    }
                }
                #endif
                
                // AO
                float ao = 1.0;
                int3 aoPos = voxelPos + int3(normal);
                int aoCount = 0;
                [unroll]
                for (int dx = -1; dx <= 1; dx++)
                {
                    [unroll]
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int3 checkPos;
                        if (abs(normal.y) > 0.5)
                            checkPos = aoPos + int3(dx, 0, dz);
                        else if (abs(normal.x) > 0.5)
                            checkPos = aoPos + int3(0, dx, dz);
                        else
                            checkPos = aoPos + int3(dx, dz, 0);
                        
                        if (IsInBounds(checkPos, _WorldSize) && GetMaterialId(ReadVoxel(checkPos)) != MAT_AIR)
                            aoCount++;
                    }
                }
                ao = 1.0 - aoCount * 0.06;
                
                // Combine lighting
                float3 ambient = _AmbientColor.rgb * ao;
                float3 diffuse = _SunColor.rgb * _SunIntensity * NdotL * shadowFactor;
                float3 specular = _SunColor.rgb * spec * shadowFactor;
                
                float3 finalColor = baseColor * (ambient + diffuse) + specular;
                
                // --- Emission from light level (propagated voxel light) ---
                // Spatially smooth light by averaging with 6 neighbors to reduce
                // discrete banding and per-frame flicker from propagation convergence.
                uint centerLight = GetLightLevel(voxelData);
                float smoothedLight = float(centerLight) * 3.0;
                float lightWeight = 3.0;
                {
                    int3 nOffsets[6] = {
                        int3(1,0,0), int3(-1,0,0),
                        int3(0,1,0), int3(0,-1,0),
                        int3(0,0,1), int3(0,0,-1)
                    };
                    [unroll]
                    for (int li = 0; li < 6; li++)
                    {
                        int3 nPos = voxelPos + nOffsets[li];
                        smoothedLight += float(GetLightLevel(ReadVoxel(nPos)));
                        lightWeight += 1.0;
                    }
                }
                float lightFactor = smoothedLight / (lightWeight * 15.0);
                if (lightFactor > 0.001)
                {
                    // Warm light tint from nearby emissive sources
                    float3 voxelLightColor = float3(1.0, 0.7, 0.3) * lightFactor * 0.8;
                    finalColor += baseColor * voxelLightColor;
                }
                
                // --- Heat glow: hot materials visually glow ---
                uint temperature = GetTemperature(voxelData);
                if (temperature > 4)
                {
                    float heatFactor = float(temperature - 4) / 11.0; // 0 to 1
                    float3 heatColor = lerp(float3(0.6, 0.1, 0.0), float3(1.5, 0.8, 0.2), heatFactor);
                    finalColor += heatColor * heatFactor * 1.5;
                }
                
                // --- Emission for lava ---
                if (matId == MAT_LAVA)
                {
                    float pulse = 0.8 + 0.2 * sin(_Time.y * 2.0 + variation * 6.28);
                    finalColor += float3(2.5, 0.8, 0.1) * pulse;
                }
                
                // --- Emission for hot coal ---
                if (matId == MAT_COAL && temperature > 4)
                {
                    float coalGlow = float(temperature - 4) / 11.0;
                    float pulse = 0.7 + 0.3 * sin(_Time.y * 1.5 + variation * 3.14);
                    finalColor += float3(1.2, 0.3, 0.05) * coalGlow * pulse;
                }
                
                return finalColor;
            }
            
            // --- Fragment Shader ---
            
            FragOutput frag(Varyings input)
            {
                FragOutput output;
                
                // Ray setup in world space
                float3 camPos = GetCameraPositionWS();
                float3 rayDir = normalize(input.positionWS - camPos);
                
                // Transform to voxel space
                float3 voxelOrigin = WorldToVoxel(camPos);
                float3 voxelDir = rayDir;
                
                // Ray march with transparency support
                TransparentLayer transLayers[MAX_TRANSPARENT_LAYERS];
                int transCount;
                RayHit hit = RayMarchVoxels(voxelOrigin, voxelDir, _MaxSteps,
                                            transLayers, transCount);
                
                if (!hit.hit)
                {
                    output.color = _FogColor;
                    output.depth = 0.0;
                    return output;
                }
                
                float3 L = normalize(_SunDir);
                
                // --- Shade opaque background (or fog if only transparent hits) ---
                uint opaqueMatId = GetMaterialId(hit.voxelData);
                float opaqueOpacity = GetMaterialOpacity(opaqueMatId);
                
                float3 opaqueColor;
                float3 depthWorldPos;
                
                if (opaqueOpacity >= 1.0)
                {
                    // We have an opaque surface behind transparent layers
                    opaqueColor = ShadeVoxel(hit.voxelPos, hit.normal, hit.voxelData, voxelDir, L);
                    depthWorldPos = VoxelToWorld(float3(hit.voxelPos) + 0.5);
                }
                else
                {
                    // Only transparent hits, no opaque behind
                    opaqueColor = _FogColor.rgb;
                    depthWorldPos = VoxelToWorld(float3(hit.voxelPos) + 0.5);
                }
                
                // --- Composite transparent layers (back-to-front) ---
                float3 composited = opaqueColor;
                
                for (int tIdx = transCount - 1; tIdx >= 0; tIdx--)
                {
                    float3 tColor = transLayers[tIdx].color;
                    float tOpacity = transLayers[tIdx].opacity;
                    uint tVoxelData = transLayers[tIdx].voxelData;
                    uint tMatId = GetMaterialId(tVoxelData);
                    
                    // Add subtle variation to transparent layers
                    float tVar = HashToFloat(HashPos(transLayers[tIdx].voxelPos, 77u));
                    tColor *= lerp(0.93, 1.07, tVar);
                    
                    // Shade the transparent surface (simplified, no shadow)
                    float3 tN = normalize(transLayers[tIdx].normal);
                    float tNdotL = max(dot(tN, L), 0.0);
                    float3 tLit = tColor * (_AmbientColor.rgb + _SunColor.rgb * _SunIntensity * tNdotL * 0.5);
                    
                    // Light level contribution on transparent surfaces
                    uint tLight = GetLightLevel(tVoxelData);
                    if (tLight > 0)
                    {
                        float tlf = float(tLight) / 15.0;
                        tLit += tColor * float3(1.0, 0.7, 0.3) * tlf * 0.5;
                    }
                    
                    // Water refraction-like tint
                    if (tMatId == MAT_WATER)
                    {
                        composited *= lerp(float3(1, 1, 1), float3(0.7, 0.85, 1.0), tOpacity);
                    }
                    
                    // Glass: slight color tint
                    if (tMatId == MAT_GLASS)
                    {
                        // Specular on glass
                        float3 tV = normalize(-voxelDir);
                        float3 tH = normalize(L + tV);
                        float tSpec = pow(max(dot(tN, tH), 0.0), 64.0) * 0.8;
                        tLit += _SunColor.rgb * tSpec;
                    }
                    
                    // Blend: front-to-back compositing
                    composited = lerp(composited, tLit, tOpacity);
                }
                
                // --- Distance fog ---
                float dist = distance(camPos, depthWorldPos);
                float fogFactor = 1.0 - exp(-dist * _FogDensity);
                float distFade = saturate((dist - _MaxRenderDist * 0.75) / (_MaxRenderDist * 0.25));
                fogFactor = max(fogFactor, distFade);
                composited = lerp(composited, _FogColor.rgb, saturate(fogFactor));
                
                // HDR output
                output.color = float4(composited, 1.0);
                
                // Depth output
                float4 clipPos = TransformWorldToHClip(depthWorldPos);
                output.depth = clipPos.z / clipPos.w;
                
                return output;
            }
            
            ENDHLSL
        }
        
        // Depth-only pass for shadows & depth prepass
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            
            Cull [_Cull]
            ZWrite On
            ColorMask 0
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment fragDepth
            #pragma target 5.0
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "VoxelCommon.hlsl"
            
            StructuredBuffer<uint> _VoxelBuffer;
            StructuredBuffer<uint> _SVOBuffer;
            
            int _WorldSize;
            int _BrickSize;
            int _BrickMapSize;
            float _VoxelScale;
            float3 _WorldOrigin;
            int _MaxSteps;
            float _MaxRenderDist;
            
            // SVO hierarchy parameters
            int4 _SVOLevelOffsets;
            int2 _SVOLevelOffsets2;
            int _SVOLevelCount;
            
            struct Attributes { float4 positionOS : POSITION; };
            struct Varyings 
            { 
                float4 positionCS : SV_POSITION; 
                float3 positionWS : TEXCOORD0; 
            };
            
            struct RayHit
            {
                bool hit;
                int3 voxelPos;
                float3 normal;
                float t;
                uint voxelData;
            };
            
            Varyings vert(Attributes input)
            {
                Varyings o;
                o.positionWS = TransformObjectToWorld(input.positionOS.xyz);
                o.positionCS = TransformWorldToHClip(o.positionWS);
                return o;
            }
            
            float3 WorldToVoxel(float3 worldPos)
            {
                return (worldPos - _WorldOrigin) / _VoxelScale;
            }
            
            float3 VoxelToWorld(float3 voxelPos)
            {
                return voxelPos * _VoxelScale + _WorldOrigin;
            }
            
            uint ReadVoxel(int3 pos)
            {
                if (any(pos < 0) || any(pos >= _WorldSize)) return 0;
                return _VoxelBuffer[Flatten3D(pos, _WorldSize)];
            }
            
            // SVO helpers for depth pass (same logic as forward pass)
            int GetSVOLevelOffset_Depth(int level)
            {
                if (level == 0) return _SVOLevelOffsets.x;
                if (level == 1) return _SVOLevelOffsets.y;
                if (level == 2) return _SVOLevelOffsets.z;
                if (level == 3) return _SVOLevelOffsets.w;
                if (level == 4) return _SVOLevelOffsets2.x;
                return _SVOLevelOffsets2.y;
            }
            
            int TrySVOSkip_Depth(int3 pos)
            {
                int bestCellSize = 0;
                [unroll]
                for (int level = MAX_SVO_LEVELS - 1; level >= 0; level--)
                {
                    if (level >= _SVOLevelCount) continue;
                    if (bestCellSize > 0) continue;
                    
                    int cellVoxelSize = _BrickSize << level;
                    int gridSize = _WorldSize / cellVoxelSize;
                    int3 cellPos = pos / cellVoxelSize;
                    
                    if (any(cellPos < 0) || any(cellPos >= gridSize)) continue;
                    
                    int levelOffset = GetSVOLevelOffset_Depth(level);
                    if (_SVOBuffer[levelOffset + Flatten3D(cellPos, gridSize)] == 0)
                    {
                        bestCellSize = cellVoxelSize;
                    }
                }
                return bestCellSize;
            }
            
            bool IntersectAABB_Depth(float3 ro, float3 rd, float3 bmin, float3 bmax, out float tN, out float tF)
            {
                float3 inv=1.0/rd;
                float3 t0=(bmin-ro)*inv, t1=(bmax-ro)*inv;
                float3 tmn=min(t0,t1), tmx=max(t0,t1);
                tN=max(max(tmn.x,tmn.y),tmn.z);
                tF=min(min(tmx.x,tmx.y),tmx.z);
                return tN<=tF&&tF>0;
            }
            
            RayHit RayMarchDepth(float3 origin, float3 dir, int maxSteps)
            {
                RayHit result;
                result.hit = false;
                result.t = 1e10;
                result.normal = float3(0,1,0);
                result.voxelPos = int3(0,0,0);
                result.voxelData = 0;
                
                float tNear, tFar;
                if (!IntersectAABB_Depth(origin, dir, float3(0,0,0), float3(_WorldSize,_WorldSize,_WorldSize), tNear, tFar))
                    return result;
                tNear = max(tNear, 0.0001);
                
                // Max traversal distance for early termination
                float maxVoxelDist = _MaxRenderDist / max(_VoxelScale, 0.001);
                tFar = min(tFar, tNear + maxVoxelDist);
                
                float3 startPos = clamp(origin + dir * tNear, 0.001, (float)_WorldSize - 0.001);
                int3 pos = clamp(int3(floor(startPos)), int3(0,0,0), int3(_WorldSize-1,_WorldSize-1,_WorldSize-1));
                int3 stepDir = int3(0, 0, 0);
                
                float3 safeDir = dir;
                safeDir.x = abs(safeDir.x)<1e-8 ? 1e-8 : safeDir.x;
                safeDir.y = abs(safeDir.y)<1e-8 ? 1e-8 : safeDir.y;
                safeDir.z = abs(safeDir.z)<1e-8 ? 1e-8 : safeDir.z;
                stepDir = int3(
                    safeDir.x >= 0 ? 1 : -1,
                    safeDir.y >= 0 ? 1 : -1,
                    safeDir.z >= 0 ? 1 : -1
                );
                float3 invDir = 1.0/safeDir;
                float3 tDelta = abs(invDir);
                
                float3 nextBound;
                nextBound.x = stepDir.x>0 ? float(pos.x+1) : float(pos.x);
                nextBound.y = stepDir.y>0 ? float(pos.y+1) : float(pos.y);
                nextBound.z = stepDir.z>0 ? float(pos.z+1) : float(pos.z);
                float3 tMax = (nextBound - origin) * invDir;
                float3 normal = float3(0,0,0);
                
                for (int i=0; i<maxSteps; i++)
                {
                    if (any(pos<0)||any(pos>=_WorldSize)) break;
                    
                    // Early termination based on distance
                    float curT = min(min(tMax.x, tMax.y), tMax.z);
                    if (curT > tNear + maxVoxelDist) break;
                    
                    // SVO hierarchical empty-space skip
                    int skipCellSize = TrySVOSkip_Depth(pos);
                    
                    if (skipCellSize > 0)
                    {
                        int3 cellPos = pos / skipCellSize;
                        float3 cellMinF = float3(cellPos * skipCellSize);
                        float3 cellMaxF = cellMinF + float(skipCellSize);
                        float3 tE0 = (cellMinF - origin) * invDir;
                        float3 tE1 = (cellMaxF - origin) * invDir;
                        float3 tFA = max(tE0, tE1);
                        float tCellExit = min(min(tFA.x, tFA.y), tFA.z);
                        float3 jp = origin + safeDir * (tCellExit + 0.001);
                        pos = int3(floor(jp));
                        nextBound.x = stepDir.x>0 ? float(pos.x+1) : float(pos.x);
                        nextBound.y = stepDir.y>0 ? float(pos.y+1) : float(pos.y);
                        nextBound.z = stepDir.z>0 ? float(pos.z+1) : float(pos.z);
                        tMax = (nextBound - origin) * invDir;
                        continue;
                    }
                    
                    if (GetMaterialId(ReadVoxel(pos))!=MAT_AIR)
                    {
                        result.hit=true; result.voxelPos=pos; result.normal=normal;
                        return result;
                    }
                    
                    if(tMax.x<tMax.y){
                        if(tMax.x<tMax.z){pos.x+=stepDir.x;tMax.x+=tDelta.x;normal=float3(-stepDir.x,0,0);}
                        else{pos.z+=stepDir.z;tMax.z+=tDelta.z;normal=float3(0,0,-stepDir.z);}
                    }else{
                        if(tMax.y<tMax.z){pos.y+=stepDir.y;tMax.y+=tDelta.y;normal=float3(0,-stepDir.y,0);}
                        else{pos.z+=stepDir.z;tMax.z+=tDelta.z;normal=float3(0,0,-stepDir.z);}
                    }
                }
                return result;
            }
            
            float4 fragDepth(Varyings input) : SV_Target
            {
                float3 camPos = GetCameraPositionWS();
                float3 rayDir = normalize(input.positionWS - camPos);
                float3 vo = WorldToVoxel(camPos);
                
                RayHit hit = RayMarchDepth(vo, rayDir, _MaxSteps);
                if (!hit.hit) discard;
                
                return 0;
            }
            
            ENDHLSL
        }
    }
    
    FallBack Off
}
