Shader "VoxelEngine/InstancedIndirect"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1,1,1,1)
        _BaseMap ("Base Map", 2D) = "white" {}
    }

    SubShader
    {
        Tags { "Queue"="Geometry" "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            Name "Forward"

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            StructuredBuffer<float4x4> _InstanceMatrices;

            sampler2D _BaseMap;
            float4 _BaseMap_ST;
            float4 _BaseColor;

            struct Attributes
            {
                float3 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                uint instanceID : SV_InstanceID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                float4x4 objectToWorld = _InstanceMatrices[input.instanceID];
                float4 worldPos = mul(objectToWorld, float4(input.positionOS, 1.0));
                output.positionCS = mul(UNITY_MATRIX_VP, worldPos);
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                return output;
            }

            fixed4 frag(Varyings input) : SV_Target
            {
                fixed4 texColor = tex2D(_BaseMap, input.uv);
                return texColor * _BaseColor;
            }
            ENDHLSL
        }
    }
}
