using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Marks an object as an indirect instancing source.
    /// Renderer will auto-batch sources by mesh/material/submesh/layer.
    /// </summary>
    [DisallowMultipleComponent]
    public class VoxelIndirectInstanceSource : MonoBehaviour
    {
        [SerializeField] private bool useMeshFromRenderer = true;
        [SerializeField] private Mesh meshOverride;
        [SerializeField] private Material materialOverride;
        [SerializeField] private int subMeshIndex;
        [SerializeField] [Range(-1, 31)] private int layerOverride = -1;
        [SerializeField] private bool renderEvenIfObjectDisabled;

        public bool IsRenderable(bool includeInactiveSources)
        {
            if (includeInactiveSources)
                return true;

            if (!isActiveAndEnabled)
                return renderEvenIfObjectDisabled;

            return true;
        }

        public bool TryGetDrawData(out Mesh mesh, out Material material, out int subMesh, out int layer)
        {
            mesh = meshOverride;
            material = materialOverride;
            subMesh = Mathf.Max(0, subMeshIndex);
            layer = layerOverride >= 0 ? layerOverride : gameObject.layer;

            if (useMeshFromRenderer)
            {
                if (mesh == null)
                {
                    var meshFilter = GetComponent<MeshFilter>();
                    if (meshFilter != null)
                        mesh = meshFilter.sharedMesh;
                }

                if (material == null)
                {
                    var meshRenderer = GetComponent<MeshRenderer>();
                    if (meshRenderer != null)
                        material = meshRenderer.sharedMaterial;
                }
            }

            return mesh != null && material != null;
        }
    }
}
