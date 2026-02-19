Shader "VoxelEngine/GlassDome"
{
    Properties
    {
        [Header(Glass Appearance)]
        _GlassColor ("Glass Tint", Color) = (0.72, 0.88, 0.95, 0.08)
        _GlassThicknessTint ("Thickness Tint", Color) = (0.55, 0.78, 0.85, 1.0)
        _FresnelPower ("Fresnel Power", Range(0.5, 8.0)) = 2.5
        _FresnelStrength ("Fresnel Strength", Range(0.0, 1.0)) = 0.45
        _ReflectionStrength ("Reflection Strength", Range(0.0, 1.0)) = 0.08
        _Smoothness ("Smoothness", Range(0.0, 1.0)) = 0.95
        _BaseAlpha ("Base Alpha (glass opacity)", Range(0.0, 0.3)) = 0.06

        [Header(Frame and Edges)]
        _FrameColor ("Frame Color", Color) = (0.35, 0.40, 0.45, 1.0)
        _FrameWidth ("Frame Width", Range(0.002, 0.06)) = 0.015
        _FrameAlpha ("Frame Alpha", Range(0.0, 1.0)) = 0.85
        _EdgeColor ("Edge Glow Color", Color) = (0.5, 0.8, 1.0, 1.0)
        _EdgeWidth ("Edge Width", Range(0.0, 0.08)) = 0.012
        _EdgeGlowIntensity ("Edge Glow Intensity", Range(0.0, 2.0)) = 0.5

        [Header(Grid Pattern)]
        _GridColor ("Grid Color", Color) = (0.6, 0.85, 0.95, 0.25)
        _GridLineCount ("Grid Lines Per Face", Float) = 8.0
        _GridThickness ("Grid Thickness", Range(0.002, 0.08)) = 0.018
        _GridFade ("Grid Fade Distance", Float) = 80.0

        [Header(Dome Shape)]
        _DomeCenter ("Dome Center", Vector) = (0, 0, 0, 0)
        _DomeExtent ("Dome Extent", Float) = 32.0
        _DomeMargin ("Dome Margin", Float) = 0.5

        [Header(Culling)]
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull Mode", Float) = 2
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Transparent+100"
        }

        Pass
        {
            Name "GlassDome"
            Tags { "LightMode" = "UniversalForward" }

            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #pragma multi_compile_fog
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // --- Properties ---
            float4 _GlassColor;
            float4 _GlassThicknessTint;
            float _FresnelPower;
            float _FresnelStrength;
            float _ReflectionStrength;
            float _Smoothness;
            float _BaseAlpha;

            float4 _FrameColor;
            float _FrameWidth;
            float _FrameAlpha;
            float4 _EdgeColor;
            float _EdgeWidth;
            float _EdgeGlowIntensity;

            float4 _GridColor;
            float _GridLineCount;
            float _GridThickness;
            float _GridFade;

            float3 _DomeCenter;
            float _DomeExtent;
            float _DomeMargin;

            // --- Structures ---
            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float3 viewDirWS  : TEXCOORD2;
                float  fogFactor  : TEXCOORD3;
                float3 localPos   : TEXCOORD4;
            };

            // --- Vertex ---
            Varyings vert(Attributes input)
            {
                Varyings o;
                VertexPositionInputs vpi = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs   vni = GetVertexNormalInputs(input.normalOS);

                o.positionCS = vpi.positionCS;
                o.positionWS = vpi.positionWS;
                o.normalWS   = vni.normalWS;
                o.viewDirWS  = GetWorldSpaceNormalizeViewDir(vpi.positionWS);
                o.fogFactor  = ComputeFogFactor(vpi.positionCS.z);
                // Mesh vertices go from [-margin, 1+margin].
                // Remap to [0, 1] for edge/frame/grid calculations.
                float range = 1.0 + _DomeMargin * 2.0;
                o.localPos   = (input.positionOS.xyz + _DomeMargin) / range;
                return o;
            }

            // --- Grid pattern (uses face-local UV, fixed line count per face) ---
            float GridPattern(float3 localUV, float3 normal)
            {
                // localUV is [0,1] on the face
                float3 absN = abs(normal);
                float2 uv;
                if (absN.y > absN.x && absN.y > absN.z)
                    uv = localUV.xz;
                else if (absN.x > absN.z)
                    uv = localUV.yz;
                else
                    uv = localUV.xy;

                // Fixed number of grid lines per face
                uv *= _GridLineCount;
                float2 grid = abs(frac(uv) - 0.5);
                float lines = min(grid.x, grid.y);
                return 1.0 - smoothstep(0.0, _GridThickness, lines);
            }

            // --- Metal frame at box edges (where two faces meet) ---
            float FrameFactor(float3 localPos)
            {
                // Each axis: distance to nearest edge (0 or 1)
                float3 d = min(localPos, 1.0 - localPos);
                // Count how many axes are near an edge
                float fx = step(d.x, _FrameWidth) ? 1.0 : 0.0;
                float fy = step(d.y, _FrameWidth) ? 1.0 : 0.0;
                float fz = step(d.z, _FrameWidth) ? 1.0 : 0.0;
                // Frame appears where at least 2 axes are near edges (= the 12 box edges)
                float edgeCount = fx + fy + fz;
                return saturate(edgeCount - 1.0);
            }

            // --- Subtle edge glow (proximity to any single edge) ---
            float EdgeFactor(float3 localPos)
            {
                float3 d = min(localPos, 1.0 - localPos);
                float minDist = min(min(d.x, d.y), d.z);
                return 1.0 - smoothstep(0.0, _EdgeWidth, minDist);
            }

            // --- Fragment ---
            float4 frag(Varyings input, bool isFrontFace : SV_IsFrontFace) : SV_Target
            {
                float3 N = normalize(input.normalWS) * (isFrontFace ? 1.0 : -1.0);
                float3 V = normalize(input.viewDirWS);
                float3 lp = saturate(input.localPos); // [0,1]

                // === Glass base ===
                float NdotV = saturate(dot(N, V));
                float viewGrazing = 1.0 - NdotV;
                float fresnel = pow(viewGrazing, _FresnelPower);
                float fresnelTerm = fresnel * _FresnelStrength;

                float3 baseColor = _GlassColor.rgb;
                float baseAlpha = _BaseAlpha + fresnelTerm * 0.35;

                // Thickness tint: glass gets slightly coloured at glancing angles
                baseColor = lerp(baseColor, _GlassThicknessTint.rgb, saturate(viewGrazing * 0.7));

                // === Subtle specular highlight (not sky reflection) ===
                Light mainLight = GetMainLight();
                float3 L = normalize(mainLight.direction);
                float3 H = normalize(V + L);
                float NdotH = saturate(dot(N, H));
                float specPower = lerp(64.0, 512.0, _Smoothness);
                float specular = pow(NdotH, specPower);
                // Small, sharp specular spot
                baseColor += mainLight.color.rgb * specular * 0.18 * (0.4 + fresnelTerm);
                baseAlpha = max(baseAlpha, specular * 0.18);

                // === Minimal reflection (keep it subtle, not skybox-like) ===
                float3 reflDir = reflect(-V, N);
                float reflUp = saturate(reflDir.y * 0.5 + 0.5);
                float3 reflTint = lerp(float3(0.7, 0.75, 0.8), float3(0.85, 0.9, 0.95), reflUp);
                baseColor = lerp(baseColor, reflTint, _ReflectionStrength * fresnelTerm);

                // === Grid pattern (fixed lines per face) ===
                float dist = length(input.positionWS - _WorldSpaceCameraPos);
                float gridFade = 1.0 - saturate(dist / _GridFade);
                float grid = GridPattern(lp, N) * gridFade;
                baseColor = lerp(baseColor, _GridColor.rgb, grid * _GridColor.a);
                baseAlpha = max(baseAlpha, grid * _GridColor.a * 0.6);

                // === Metal frame at box edges ===
                float frame = FrameFactor(lp);
                baseColor = lerp(baseColor, _FrameColor.rgb, frame);
                baseAlpha = lerp(baseAlpha, _FrameAlpha, frame);

                // === Subtle edge glow ===
                float edge = EdgeFactor(lp);
                float edgeBlend = edge * _EdgeGlowIntensity * (1.0 - frame); // don't add glow on frame
                baseColor += _EdgeColor.rgb * edgeBlend * 0.3;
                baseAlpha = max(baseAlpha, edgeBlend * _EdgeColor.a * 0.35);

                // === Final ===
                baseAlpha = saturate(baseAlpha);
                float3 finalColor = MixFog(baseColor, input.fogFactor);

                return float4(finalColor, baseAlpha);
            }
            ENDHLSL
        }
    }

    Fallback Off
}
