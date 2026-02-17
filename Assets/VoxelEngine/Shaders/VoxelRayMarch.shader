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
            
            RayHit RayMarchVoxels(float3 origin, float3 dir, int maxSteps)
            {
                RayHit result;
                result.hit = false;
                result.t = 1e10;
                result.normal = float3(0, 1, 0);
                result.voxelPos = int3(0, 0, 0);
                result.voxelData = 0;
                
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
                
                // Step direction
                int3 stepDir = int3(sign(dir));
                
                // Prevent division by zero
                float3 safeDir = dir;
                safeDir.x = abs(safeDir.x) < 1e-8 ? (safeDir.x >= 0 ? 1e-8 : -1e-8) : safeDir.x;
                safeDir.y = abs(safeDir.y) < 1e-8 ? (safeDir.y >= 0 ? 1e-8 : -1e-8) : safeDir.y;
                safeDir.z = abs(safeDir.z) < 1e-8 ? (safeDir.z >= 0 ? 1e-8 : -1e-8) : safeDir.z;
                
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
                    
                    // Brick map acceleration
                    int3 brickPos = pos / _BrickSize;
                    if (IsBrickEmpty(brickPos))
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
                        pos = clamp(pos, int3(0,0,0), int3(_WorldSize-1, _WorldSize-1, _WorldSize-1));
                        
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
                        result.hit = true;
                        result.voxelPos = pos;
                        result.normal = normal;
                        result.voxelData = voxel;
                        
                        // Compute hit distance
                        float3 tLast = tMax - tDelta;
                        result.t = max(max(tLast.x * (abs(normal.x) > 0.5 ? 1 : 0) + 
                                           tLast.y * (abs(normal.y) > 0.5 ? 1 : 0) + 
                                           tLast.z * (abs(normal.z) > 0.5 ? 1 : 0), 0), tNear);
                        
                        // Fallback distance computation
                        if (length(normal) < 0.5)
                        {
                            result.t = tNear;
                            result.normal = float3(0, 1, 0);
                        }
                        
                        return result;
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
                
                return result;
            }
            
            // --- Shadow Ray (simplified, no brick skip for perf) ---
            
            bool CastShadowRay(float3 origin, float3 dir, int maxSteps)
            {
                int3 pos = int3(floor(origin));
                int3 stepDir = int3(sign(dir));
                
                float3 safeDir = dir;
                safeDir.x = abs(safeDir.x) < 1e-8 ? 1e-8 : safeDir.x;
                safeDir.y = abs(safeDir.y) < 1e-8 ? 1e-8 : safeDir.y;
                safeDir.z = abs(safeDir.z) < 1e-8 ? 1e-8 : safeDir.z;
                
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
                    
                    // Brick map skip for shadow rays too
                    int3 brickPos = pos / _BrickSize;
                    if (IsBrickEmpty(brickPos))
                    {
                        float3 brickMinF = float3(brickPos * _BrickSize);
                        float3 brickMaxF = brickMinF + float(_BrickSize);
                        float3 tExit0 = (brickMinF - origin) * invDir;
                        float3 tExit1 = (brickMaxF - origin) * invDir;
                        float3 tFarAxis = max(tExit0, tExit1);
                        float tBrickExit = min(min(tFarAxis.x, tFarAxis.y), tFarAxis.z);
                        
                        float3 jumpPos = origin + safeDir * (tBrickExit + 0.001);
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
            
            // --- Fragment Shader ---
            
            FragOutput frag(Varyings input)
            {
                FragOutput output;
                
                // Ray setup in world space
                float3 camPos = GetCameraPositionWS();
                float3 rayDir = normalize(input.positionWS - camPos);
                
                // Transform to voxel space
                float3 voxelOrigin = WorldToVoxel(camPos);
                float3 voxelDir = rayDir; // Direction unchanged by translation+uniform scale
                
                // Ray march
                RayHit hit = RayMarchVoxels(voxelOrigin, voxelDir, _MaxSteps);
                
                if (!hit.hit)
                    discard;
                
                // --- Shading ---
                uint matId = GetMaterialId(hit.voxelData);
                uint color565 = GetColor565(hit.voxelData);
                
                // Base color
                float3 baseColor;
                if (color565 != 0)
                    baseColor = Color565ToRGB(color565);
                else
                    baseColor = GetDefaultMaterialColor(matId);
                
                // Add subtle variation based on position
                float variation = HashToFloat(HashPos(hit.voxelPos, 42u));
                baseColor *= lerp(0.9, 1.1, variation);
                
                float3 N = normalize(hit.normal);
                float3 L = normalize(_SunDir);
                float3 V = normalize(-voxelDir);
                float3 H = normalize(L + V);
                
                // Diffuse (Lambert)
                float NdotL = max(dot(N, L), 0.0);
                
                // Specular (Blinn-Phong)
                float NdotH = max(dot(N, H), 0.0);
                float specIntensity = (matId == MAT_WATER || matId == MAT_GLASS) ? 0.6 :
                                      (matId == MAT_LAVA) ? 0.0 :
                                      (matId == MAT_IRON || matId == MAT_GOLD) ? 0.4 : 0.08;
                float spec = pow(NdotH, _SpecularPower) * specIntensity;
                
                // Shadow ray
                float shadowFactor = 1.0;
                #ifdef VOXEL_SHADOWS_ON
                {
                    float3 shadowOrigin = float3(hit.voxelPos) + 0.5 + hit.normal * 1.01;
                    if (CastShadowRay(shadowOrigin, L, _MaxShadowSteps))
                        shadowFactor = 0.25;
                }
                #endif
                
                // Simple AO: check adjacent voxels in normal direction
                float ao = 1.0;
                int3 aoPos = hit.voxelPos + int3(hit.normal);
                int aoCount = 0;
                [unroll]
                for (int dx = -1; dx <= 1; dx++)
                {
                    [unroll]
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int3 checkPos;
                        if (abs(hit.normal.y) > 0.5)
                            checkPos = aoPos + int3(dx, 0, dz);
                        else if (abs(hit.normal.x) > 0.5)
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
                
                // Emission for lava
                if (matId == MAT_LAVA)
                {
                    float pulse = 0.8 + 0.2 * sin(_Time.y * 2.0 + variation * 6.28);
                    finalColor += float3(2.5, 0.8, 0.1) * pulse;
                }
                
                // Distance fog
                float3 hitWorldPos = VoxelToWorld(float3(hit.voxelPos) + 0.5);
                float dist = distance(camPos, hitWorldPos);
                float fogFactor = 1.0 - exp(-dist * _FogDensity);
                
                // Render distance fade: smooth transition to fog at max distance
                float distFade = saturate((dist - _MaxRenderDist * 0.75) / (_MaxRenderDist * 0.25));
                fogFactor = max(fogFactor, distFade);
                
                finalColor = lerp(finalColor, _FogColor.rgb, saturate(fogFactor));
                
                // HDR output
                output.color = float4(finalColor, 1.0);
                
                // Depth output
                float4 clipPos = TransformWorldToHClip(hitWorldPos);
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
                int3 stepDir = int3(sign(dir));
                
                float3 safeDir = dir;
                safeDir.x = abs(safeDir.x)<1e-8 ? 1e-8 : safeDir.x;
                safeDir.y = abs(safeDir.y)<1e-8 ? 1e-8 : safeDir.y;
                safeDir.z = abs(safeDir.z)<1e-8 ? 1e-8 : safeDir.z;
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
                    if (IsBrickEmpty(bp))
                    {
                        float3 bMinF=float3(bp*_BrickSize), bMaxF=bMinF+float(_BrickSize);
                        float3 te0=(bMinF-origin)*invDir, te1=(bMaxF-origin)*invDir;
                        float tBE=min(min(max(te0.x,te1.x),max(te0.y,te1.y)),max(te0.z,te1.z));
                        float3 jp=origin+safeDir*(tBE+0.001);
                        pos=clamp(int3(floor(jp)),int3(0,0,0),int3(_WorldSize-1,_WorldSize-1,_WorldSize-1));
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
