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
        _FastLightingMaxDist ("Fast Lighting Max Distance", Float) = 40
        _ShadowRayMaxDist ("Shadow Ray Max Distance", Float) = 32

        [Header(Heat Radiation)]
        _HeatHazeStrength ("Heat Haze Strength", Range(0.0, 0.03)) = 0.007
        _HeatHazeScale ("Heat Haze Noise Scale", Float) = 0.25
        _HeatHazeSpeed ("Heat Haze Speed", Float) = 1.2
        _HeatHazeStartTemp ("Heat Haze Start Temp", Range(0, 15)) = 6
        _HeatHazeSampleStep ("Heat Haze Sample Step", Float) = 1.6
        _HeatRadiationStartTemp ("Heat Radiation Start Temp", Range(0, 15)) = 5
        _HeatRadiationIntensity ("Heat Radiation Intensity", Range(0, 4)) = 1.8
        
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
            
            Cull Off
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
            StructuredBuffer<uint> _BrickMap;
            
            // --- Parameters (set from C#) ---
            int _WorldSize;
            int _BrickSize;
            int _BrickMapSize;
            float _VoxelScale;
            float3 _WorldOrigin;   // World-space origin of the voxel volume
            int _MaxSteps;
            int _MaxShadowSteps;
            
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
            float _FastLightingMaxDist;
            float _ShadowRayMaxDist;
            
            // Shadow blending
            float _ShadowStrength;

            // Heat haze + blackbody radiation
            float _HeatHazeStrength;
            float _HeatHazeScale;
            float _HeatHazeSpeed;
            float _HeatHazeStartTemp;
            float _HeatHazeSampleStep;
            float _HeatRadiationStartTemp;
            float _HeatRadiationIntensity;
            
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
            
            // Check whether the camera is inside the voxel volume AABB.
            // Margin must be >= the box mesh margin (0.12 in object space = 0.12 * _WorldSize voxels)
            // so we switch to "render all faces" mode before the camera reaches the mesh surface.
            bool IsCameraInsideVolume(float3 camVoxelPos)
            {
                float margin = 0.15 * _WorldSize;
                return all(camVoxelPos > -margin) && all(camVoxelPos < _WorldSize + margin);
            }
            
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
            
            bool IsBrickEmpty(int3 brickPos)
            {
                if (any(brickPos < 0) || any(brickPos >= _BrickMapSize))
                    return true;
                return _BrickMap[Flatten3D(brickPos, _BrickMapSize)] == 0;
            }
            
            // Check a 2x2x2 super-brick (4 bricks along each axis combined)
            // Returns true if all 8 child bricks are empty — allows larger jumps
            bool IsSuperBrickEmpty(int3 superPos)
            {
                int3 baseBrick = superPos * 2;
                [unroll]
                for (int z = 0; z < 2; z++)
                [unroll]
                for (int y = 0; y < 2; y++)
                [unroll]
                for (int x = 0; x < 2; x++)
                {
                    int3 bp = baseBrick + int3(x, y, z);
                    if (!IsBrickEmpty(bp))
                        return false;
                }
                return true;
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
                    
                    // Hierarchical empty-space skip: try super-brick (2x2x2 bricks) first
                    int3 brickPos = pos / _BrickSize;
                    int3 superPos = brickPos / 2;
                    int superBrickVoxelSize = _BrickSize * 2;
                    
                    if (IsSuperBrickEmpty(superPos))
                    {
                        // Skip entire 2x2x2 super-brick region
                        float3 sbMinF = float3(superPos * superBrickVoxelSize);
                        float3 sbMaxF = sbMinF + float(superBrickVoxelSize);
                        
                        float3 tExit0 = (sbMinF - origin) * invDir;
                        float3 tExit1 = (sbMaxF - origin) * invDir;
                        float3 tFarAxis = max(tExit0, tExit1);
                        float tSBExit = min(min(tFarAxis.x, tFarAxis.y), tFarAxis.z);
                        
                        if (tSBExit == tFarAxis.x) normal = float3(-stepDir.x, 0, 0);
                        else if (tSBExit == tFarAxis.y) normal = float3(0, -stepDir.y, 0);
                        else normal = float3(0, 0, -stepDir.z);
                        
                        float tJump = tSBExit + 0.001;
                        float3 jumpPos = origin + safeDir * tJump;
                        pos = int3(floor(jumpPos));
                        
                        nextBound.x = stepDir.x > 0 ? float(pos.x + 1) : float(pos.x);
                        nextBound.y = stepDir.y > 0 ? float(pos.y + 1) : float(pos.y);
                        nextBound.z = stepDir.z > 0 ? float(pos.z + 1) : float(pos.z);
                        tMax = (nextBound - origin) * invDir;
                        
                        continue;
                    }
                    else if (IsBrickEmpty(brickPos))
                    {
                        // Skip entire brick - compute ray exit from this brick
                        float3 brickMinF = float3(brickPos * _BrickSize);
                        float3 brickMaxF = brickMinF + float(_BrickSize);
                        
                        float3 tExit0 = (brickMinF - origin) * invDir;
                        float3 tExit1 = (brickMaxF - origin) * invDir;
                        float3 tFarAxis = max(tExit0, tExit1);
                        float tBrickExit = min(min(tFarAxis.x, tFarAxis.y), tFarAxis.z);
                        
                        // Determine exit normal
                        if (tBrickExit == tFarAxis.x) normal = float3(-stepDir.x, 0, 0);
                        else if (tBrickExit == tFarAxis.y) normal = float3(0, -stepDir.y, 0);
                        else normal = float3(0, 0, -stepDir.z);
                        
                        // Jump past the brick
                        float tJump = tBrickExit + 0.001;
                        float3 jumpPos = origin + safeDir * tJump;
                        pos = int3(floor(jumpPos));
                        
                        // Recompute tMax from new position
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
                
                // Limit shadow rays to configured distance (in voxel space)
                float maxShadowVoxelDist = _ShadowRayMaxDist / max(_VoxelScale, 0.001);
                
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
                    
                    // Super-brick skip for shadow rays too
                    int3 brickPos = pos / _BrickSize;
                    int3 sBrickPos = brickPos / 2;
                    int sBrickVS = _BrickSize * 2;
                    
                    if (IsSuperBrickEmpty(sBrickPos))
                    {
                        float3 sbMin = float3(sBrickPos * sBrickVS);
                        float3 sbMax = sbMin + float(sBrickVS);
                        float3 tE0 = (sbMin - origin) * invDir;
                        float3 tE1 = (sbMax - origin) * invDir;
                        float3 tFA = max(tE0, tE1);
                        float tSBE = min(min(tFA.x, tFA.y), tFA.z);
                        
                        float tJump = tSBE + 0.001;
                        float3 jumpPos = origin + safeDir * tJump;
                        pos = int3(floor(jumpPos));
                        
                        nextBound.x = stepDir.x > 0 ? float(pos.x + 1) : float(pos.x);
                        nextBound.y = stepDir.y > 0 ? float(pos.y + 1) : float(pos.y);
                        nextBound.z = stepDir.z > 0 ? float(pos.z + 1) : float(pos.z);
                        tMax = (nextBound - origin) * invDir;
                        continue;
                    }
                    else if (IsBrickEmpty(brickPos))
                    {
                        float3 brickMinF = float3(brickPos * _BrickSize);
                        float3 brickMaxF = brickMinF + float(_BrickSize);
                        float3 tExit0 = (brickMinF - origin) * invDir;
                        float3 tExit1 = (brickMaxF - origin) * invDir;
                        float3 tFarAxis = max(tExit0, tExit1);
                        float tBrickExit = min(min(tFarAxis.x, tFarAxis.y), tFarAxis.z);
                        
                        float tJump = tBrickExit + 0.001;
                        float3 jumpPos = origin + safeDir * tJump;
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

            float3 BlackbodyFromHeat(float heatNorm)
            {
                heatNorm = saturate(heatNorm);
                float3 c0 = float3(0.35, 0.03, 0.00);
                float3 c1 = float3(0.95, 0.16, 0.02);
                float3 c2 = float3(1.30, 0.55, 0.08);
                float3 c3 = float3(1.95, 1.35, 0.65);

                float3 mid = (heatNorm < 0.5)
                    ? lerp(c0, c1, heatNorm * 2.0)
                    : lerp(c1, c2, (heatNorm - 0.5) * 2.0);

                return lerp(mid, c3, smoothstep(0.72, 1.0, heatNorm));
            }

            float3 ApplyHeatDistortion(float3 origin, float3 dir)
            {
                if (_HeatHazeStrength <= 0.0001)
                    return dir;

                float hazeAccum = 0.0;
                float maxTempRange = max(15.0 - _HeatHazeStartTemp, 0.001);

                [unroll]
                for (int i = 1; i <= 3; i++)
                {
                    float3 samplePos = origin + dir * (_HeatHazeSampleStep * i);
                    int3 sampleVoxel = int3(floor(samplePos));
                    if (!IsInBounds(sampleVoxel, _WorldSize))
                        continue;

                    uint sampleData = ReadVoxel(sampleVoxel);
                    uint sampleMat = GetMaterialId(sampleData);
                    if (sampleMat == MAT_AIR)
                        continue;

                    float sampleTemp = (float)GetTemperature(sampleData);
                    hazeAccum += saturate((sampleTemp - _HeatHazeStartTemp) / maxTempRange);
                }

                float haze = hazeAccum / 3.0;
                if (haze <= 0.0001)
                    return dir;

                float3 upRef = abs(dir.y) > 0.92 ? float3(1, 0, 0) : float3(0, 1, 0);
                float3 right = normalize(cross(upRef, dir));
                float3 up = normalize(cross(dir, right));

                float t = _Time.y * _HeatHazeSpeed;
                float3 np = (origin + dir * (_HeatHazeSampleStep * 1.5)) * _HeatHazeScale;
                float n1 = sin(dot(np, float3(12.9898, 78.233, 37.719)) + t * 1.17);
                float n2 = cos(dot(np.zxy, float3(39.346, 11.135, 83.155)) + t * 1.41);

                float distortion = _HeatHazeStrength * haze;
                return normalize(dir + (right * n1 + up * n2) * distortion);
            }
            
            // --- Shade a single voxel surface ---
            
            float3 ShadeVoxel(int3 voxelPos, float3 normal, uint voxelData,
                              float3 voxelDir, float3 L, float viewDist)
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
                // Smooth transition for lighting detail instead of hard cutoff
                // to eliminate flickering at the shadow/lighting boundary
                float fadeDist = max(_FastLightingMaxDist, 1.0);
                float fadeStart = fadeDist * 0.75;
                float lightingDetail = 1.0 - saturate((viewDist - fadeStart) / max(fadeDist - fadeStart, 0.01));
                
                // Diffuse
                float NdotL = max(dot(N, L), 0.0);
                
                // Specular
                float NdotH = max(dot(N, H), 0.0);
                float specIntensity = (matId == MAT_WATER || matId == MAT_GLASS) ? 0.6 :
                                      (matId == MAT_LAVA) ? 0.0 :
                                      (matId == MAT_IRON || matId == MAT_GOLD) ? 0.4 : 0.08;
                float spec = pow(NdotH, _SpecularPower) * specIntensity;
                
                // Shadow with smooth distance fade
                float shadowFactor = 1.0;
                #ifdef VOXEL_SHADOWS_ON
                {
                    if (lightingDetail > 0.01 && _ShadowStrength > 0.001 && NdotL > 0.001)
                    {
                        float3 shadowOrigin = float3(voxelPos) + 0.5 + normal * 1.2 + L * 0.15;
                        if (CastShadowRay(shadowOrigin, L, _MaxShadowSteps))
                        {
                            float effectiveStrength = _ShadowStrength * lightingDetail;
                            float shadowDarkness = lerp(1.0, 0.25, effectiveStrength);
                            shadowFactor = shadowDarkness;
                        }
                    }
                }
                #endif
                
                // AO with smooth distance fade
                float ao = 1.0;
                if (lightingDetail > 0.01)
                {
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
                    ao = lerp(1.0, 1.0 - aoCount * 0.06, lightingDetail);
                }
                
                // Combine lighting
                float3 ambient = _AmbientColor.rgb * ao;
                float3 diffuse = _SunColor.rgb * _SunIntensity * NdotL * shadowFactor;
                float3 specular = _SunColor.rgb * spec * shadowFactor;
                
                float3 finalColor = baseColor * (ambient + diffuse) + specular;
                
                // --- Emission from light level (propagated voxel light) ---
                // Spatially smooth light by averaging with 6 neighbors to reduce
                // discrete banding and per-frame flicker from propagation convergence.
                uint centerLight = GetLightLevel(voxelData);
                float smoothedLight = float(centerLight) * 6.0;
                float lightWeight = 6.0;
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
                        smoothedLight += float(GetLightLevel(ReadVoxel(nPos))) * 0.8;
                        lightWeight += 0.8;
                    }
                }
                float lightFactor = smoothedLight / (lightWeight * 15.0);
                lightFactor = pow(saturate(lightFactor), 0.72);
                if (lightFactor > 0.001)
                {
                    // Warm light tint from nearby emissive sources
                    float3 voxelLightColor = float3(1.0, 0.72, 0.35) * lightFactor * 1.1;
                    finalColor += baseColor * voxelLightColor * 0.75;
                    finalColor += voxelLightColor * 0.35;
                }
                
                // --- Heat glow: hot materials visually glow ---
                uint temperature = GetTemperature(voxelData);
                if ((float)temperature > _HeatRadiationStartTemp)
                {
                    float maxTempRange = max(15.0 - _HeatRadiationStartTemp, 0.001);
                    float heatNorm = saturate(((float)temperature - _HeatRadiationStartTemp) / maxTempRange);
                    float pulse = 0.88 + 0.12 * sin(_Time.y * 2.2 + variation * 5.7);
                    float3 heatColor = BlackbodyFromHeat(heatNorm);
                    finalColor += heatColor * heatNorm * _HeatRadiationIntensity * pulse;
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
            
            FragOutput frag(Varyings input, bool isFrontFace : SV_IsFrontFace)
            {
                FragOutput output;
                
                // Ray setup in world space
                float3 camPos = GetCameraPositionWS();
                float3 rayDir = normalize(input.positionWS - camPos);
                
                // When camera is OUTSIDE the volume, discard front-face fragments
                // to avoid double ray marching (back face handles it).
                // When camera is INSIDE, render ALL faces so nothing is culled.
                float3 camVoxelPos = WorldToVoxel(camPos);
                if (!IsCameraInsideVolume(camVoxelPos) && isFrontFace)
                    discard;
                
                // Transform to voxel space
                float3 voxelOrigin = WorldToVoxel(camPos);
                float3 voxelDir = ApplyHeatDistortion(voxelOrigin, rayDir);
                
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
                    opaqueColor = ShadeVoxel(hit.voxelPos, hit.normal, hit.voxelData, voxelDir, L,
                                             distance(camPos, VoxelToWorld(float3(hit.voxelPos) + 0.5)));
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
                        float tlf = pow(saturate(float(tLight) / 15.0), 0.75);
                        tLit += tColor * float3(1.0, 0.72, 0.35) * tlf * 0.75;
                        tLit += float3(1.0, 0.72, 0.35) * tlf * 0.15;
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
            
            Cull Off
            ZWrite On
            ColorMask 0
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment fragDepth
            #pragma target 5.0
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "VoxelCommon.hlsl"
            
            StructuredBuffer<uint> _VoxelBuffer;
            StructuredBuffer<uint> _BrickMap;
            
            int _WorldSize;
            int _BrickSize;
            int _BrickMapSize;
            float _VoxelScale;
            float3 _WorldOrigin;
            int _MaxSteps;
            float _MaxRenderDist;
            
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
            
            bool IsBrickEmpty(int3 bp)
            {
                if (any(bp < 0) || any(bp >= _BrickMapSize)) return true;
                return _BrickMap[Flatten3D(bp, _BrickMapSize)] == 0;
            }
            
            bool IsSuperBrickEmpty(int3 sbp)
            {
                for (int dz = 0; dz < 2; dz++)
                for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                {
                    int3 bp = sbp * 2 + int3(dx, dy, dz);
                    if (!IsBrickEmpty(bp)) return false;
                }
                return true;
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
                    
                    int3 bp = pos/_BrickSize;
                    int3 sbp = bp / 2;
                    int sBrickVS = _BrickSize * 2;
                    
                    if (IsSuperBrickEmpty(sbp))
                    {
                        float3 sbMin = float3(sbp * sBrickVS);
                        float3 sbMax = sbMin + float(sBrickVS);
                        float3 tE0 = (sbMin - origin) * invDir;
                        float3 tE1 = (sbMax - origin) * invDir;
                        float3 tFA = max(tE0, tE1);
                        float tSBE = min(min(tFA.x, tFA.y), tFA.z);
                        float tJump = tSBE + 0.001;
                        float3 jp = origin + safeDir * tJump;
                        pos = int3(floor(jp));
                        nextBound.x = stepDir.x>0 ? float(pos.x+1) : float(pos.x);
                        nextBound.y = stepDir.y>0 ? float(pos.y+1) : float(pos.y);
                        nextBound.z = stepDir.z>0 ? float(pos.z+1) : float(pos.z);
                        tMax = (nextBound - origin) * invDir;
                        continue;
                    }
                    else if (IsBrickEmpty(bp))
                    {
                        float3 bMinF=float3(bp*_BrickSize), bMaxF=bMinF+float(_BrickSize);
                        float3 te0=(bMinF-origin)*invDir, te1=(bMaxF-origin)*invDir;
                        float tBE=min(min(max(te0.x,te1.x),max(te0.y,te1.y)),max(te0.z,te1.z));
                        float tJump=tBE+0.001;
                        float3 jp=origin+safeDir*tJump;
                        pos=int3(floor(jp));
                        nextBound.x=stepDir.x>0?float(pos.x+1):float(pos.x);
                        nextBound.y=stepDir.y>0?float(pos.y+1):float(pos.y);
                        nextBound.z=stepDir.z>0?float(pos.z+1):float(pos.z);
                        tMax=(nextBound-origin)*invDir;
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
            
            // Check whether the camera is inside the voxel volume AABB.
            // Must match the margin in the main pass.
            bool IsCameraInsideVolume_Depth(float3 camVoxelPos)
            {
                float margin = 0.15 * _WorldSize;
                return all(camVoxelPos > -margin) && all(camVoxelPos < _WorldSize + margin);
            }
            
            float4 fragDepth(Varyings input, bool isFrontFace : SV_IsFrontFace) : SV_Target
            {
                float3 camPos = GetCameraPositionWS();
                float3 vo = WorldToVoxel(camPos);
                
                // Same front-face discard optimization as main pass
                if (!IsCameraInsideVolume_Depth(vo) && isFrontFace)
                    discard;
                
                float3 rayDir = normalize(input.positionWS - camPos);
                
                RayHit hit = RayMarchDepth(vo, rayDir, _MaxSteps);
                if (!hit.hit) discard;
                
                return 0;
            }
            
            ENDHLSL
        }
    }
    
    FallBack Off
}
