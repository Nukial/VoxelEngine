using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Creates and manages a glass dome (box enclosure) around the entire voxel world.
    /// The dome mesh uses the SAME coordinate system as VoxelRenderer's box mesh:
    ///   vertices [0, 1] in local space, positioned at WorldOrigin, scaled by WorldExtent.
    /// A small margin pushes the glass slightly outside the voxel volume.
    ///
    /// Glass is rendered double-sided (Cull Off) to avoid disappearing edges/faces
    /// when camera angle changes or when transitioning between outside/inside views.
    /// </summary>
    public class VoxelGlassDome : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private VoxelWorld voxelWorld;
        [SerializeField] private VoxelDualCamera dualCamera;
        [SerializeField] private Shader glassDomeShader;

        [Header("Glass Appearance")]
        [SerializeField] private Color glassColor = new Color(0.72f, 0.88f, 0.95f, 0.08f);
        [SerializeField] private Color glassThicknessTint = new Color(0.55f, 0.78f, 0.85f, 1f);
        [SerializeField] [Range(0.5f, 8f)] private float fresnelPower = 2.5f;
        [SerializeField] [Range(0f, 1f)] private float fresnelStrength = 0.62f;
        [SerializeField] [Range(0f, 1f)] private float reflectionStrength = 0.14f;
        [SerializeField] [Range(0f, 1f)] private float smoothness = 1.0f;
        [SerializeField] [Range(0f, 0.3f)] private float baseAlpha = 0.035f;

        [Header("Frame and Edges")]
        [SerializeField] private Color frameColor = new Color(0.42f, 0.47f, 0.52f, 1f);
        [SerializeField] [Range(0.002f, 0.06f)] private float frameWidth = 0.007f;
        [SerializeField] [Range(0f, 1f)] private float frameAlpha = 0.45f;
        [SerializeField] private Color edgeColor = new Color(0.5f, 0.8f, 1f, 1f);
        [SerializeField] [Range(0f, 0.08f)] private float edgeWidth = 0.016f;
        [SerializeField] [Range(0f, 2f)] private float edgeGlowIntensity = 0.75f;

        [Header("Grid")]
        [SerializeField] private Color gridColor = new Color(0.6f, 0.85f, 0.95f, 0.08f);
        [SerializeField] private float gridLineCount = 8f;
        [SerializeField] [Range(0.002f, 0.08f)] private float gridThickness = 0.01f;

        [Header("Dome Settings")]
        [Tooltip("Normalized margin outside the voxel volume (0.02 = 2% of extent)")]
        [SerializeField] [Range(0.01f, 0.1f)] private float normalizedMargin = 0.02f;
        [SerializeField] private bool visibleInGodView = true;
        [SerializeField] private bool visibleInInsideView = true;

        // --- Runtime ---
        private Material _domeMaterial;
        private GameObject _domeObject;
        private MeshFilter _meshFilter;
        private MeshRenderer _meshRenderer;

        private static readonly int PropGlassColor = Shader.PropertyToID("_GlassColor");
        private static readonly int PropGlassThicknessTint = Shader.PropertyToID("_GlassThicknessTint");
        private static readonly int PropFresnelPower = Shader.PropertyToID("_FresnelPower");
        private static readonly int PropFresnelStrength = Shader.PropertyToID("_FresnelStrength");
        private static readonly int PropReflectionStrength = Shader.PropertyToID("_ReflectionStrength");
        private static readonly int PropSmoothness = Shader.PropertyToID("_Smoothness");
        private static readonly int PropBaseAlpha = Shader.PropertyToID("_BaseAlpha");
        private static readonly int PropFrameColor = Shader.PropertyToID("_FrameColor");
        private static readonly int PropFrameWidth = Shader.PropertyToID("_FrameWidth");
        private static readonly int PropFrameAlpha = Shader.PropertyToID("_FrameAlpha");
        private static readonly int PropEdgeColor = Shader.PropertyToID("_EdgeColor");
        private static readonly int PropEdgeWidth = Shader.PropertyToID("_EdgeWidth");
        private static readonly int PropEdgeGlowIntensity = Shader.PropertyToID("_EdgeGlowIntensity");
        private static readonly int PropGridColor = Shader.PropertyToID("_GridColor");
        private static readonly int PropGridLineCount = Shader.PropertyToID("_GridLineCount");
        private static readonly int PropGridThickness = Shader.PropertyToID("_GridThickness");
        private static readonly int PropDomeCenter = Shader.PropertyToID("_DomeCenter");
        private static readonly int PropDomeExtent = Shader.PropertyToID("_DomeExtent");
        private static readonly int PropDomeMargin = Shader.PropertyToID("_DomeMargin");
        private static readonly int PropCull = Shader.PropertyToID("_Cull");

        // =================================================================
        // Lifecycle
        // =================================================================

        private void OnEnable()
        {
            if (voxelWorld == null)
                voxelWorld = GetComponent<VoxelWorld>();
            if (voxelWorld == null)
                voxelWorld = FindFirstObjectByType<VoxelWorld>();

            if (dualCamera == null)
                dualCamera = FindFirstObjectByType<VoxelDualCamera>();

            CreateDome();
        }

        private void OnDisable()
        {
            DestroyDome();
        }

        private void Update()
        {
            if (_domeMaterial == null || voxelWorld == null) return;

            // Keep dome transform in sync with VoxelWorld every frame
            SyncTransform();
            UpdateMaterialProperties();
            UpdateVisibility();
        }

        // =================================================================
        // Dome Creation
        // =================================================================

        private void CreateDome()
        {
            if (voxelWorld == null)
            {
                Debug.LogWarning("[VoxelGlassDome] No VoxelWorld found!");
                return;
            }

            // Find or create shader
            Shader shader = glassDomeShader;
            if (shader == null)
            {
                shader = Shader.Find("VoxelEngine/GlassDome");
                if (shader == null)
                {
                    Debug.LogError("[VoxelGlassDome] GlassDome shader not found!");
                    return;
                }
            }

            // Create material
            _domeMaterial = new Material(shader);
            _domeMaterial.name = "GlassDome (Runtime)";

            // Create dome as a ROOT object (NOT child of VoxelWorld)
            // because VoxelWorld already has localScale = worldExtent,
            // and parenting would multiply scales incorrectly.
            _domeObject = new GameObject("_GlassDome");
            _domeObject.layer = gameObject.layer;

            // Mesh filter + renderer
            _meshFilter = _domeObject.AddComponent<MeshFilter>();
            _meshRenderer = _domeObject.AddComponent<MeshRenderer>();

            _meshFilter.sharedMesh = CreateDomeMesh(normalizedMargin);
            _meshRenderer.sharedMaterial = _domeMaterial;
            _meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _meshRenderer.receiveShadows = false;

            // Match VoxelWorld's transform exactly:
            //   position = WorldOrigin, scale = WorldExtent
            // The mesh vertices use [0,1] coordinates (same as VoxelRenderer),
            // with a small margin pushing glass slightly outside.
            SyncTransform();
            UpdateMaterialProperties();
        }

        /// <summary>
        /// Keep dome position and scale perfectly matched to the voxel volume.
        /// Same coordinate system as VoxelRenderer's box mesh.
        /// </summary>
        private void SyncTransform()
        {
            if (_domeObject == null || voxelWorld == null) return;

            float extent = voxelWorld.WorldExtent;
            _domeObject.transform.position = voxelWorld.WorldOrigin;
            _domeObject.transform.localScale = Vector3.one * extent;
        }

        private void DestroyDome()
        {
            if (_domeObject != null)
            {
                if (Application.isPlaying)
                    Destroy(_domeObject);
                else
                    DestroyImmediate(_domeObject);
            }

            if (_domeMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(_domeMaterial);
                else
                    DestroyImmediate(_domeMaterial);
            }

            _domeMaterial = null;
            _domeObject = null;
            _meshFilter = null;
            _meshRenderer = null;
        }

        // =================================================================
        // Material Updates
        // =================================================================

        private void UpdateMaterialProperties()
        {
            _domeMaterial.SetColor(PropGlassColor, glassColor);
            _domeMaterial.SetColor(PropGlassThicknessTint, glassThicknessTint);
            _domeMaterial.SetFloat(PropFresnelPower, fresnelPower);
            _domeMaterial.SetFloat(PropFresnelStrength, fresnelStrength);
            _domeMaterial.SetFloat(PropReflectionStrength, reflectionStrength);
            _domeMaterial.SetFloat(PropSmoothness, smoothness);
            _domeMaterial.SetFloat(PropBaseAlpha, baseAlpha);
            _domeMaterial.SetColor(PropFrameColor, frameColor);
            _domeMaterial.SetFloat(PropFrameWidth, frameWidth);
            _domeMaterial.SetFloat(PropFrameAlpha, frameAlpha);
            _domeMaterial.SetColor(PropEdgeColor, edgeColor);
            _domeMaterial.SetFloat(PropEdgeWidth, edgeWidth);
            _domeMaterial.SetFloat(PropEdgeGlowIntensity, edgeGlowIntensity);
            _domeMaterial.SetColor(PropGridColor, gridColor);
            _domeMaterial.SetFloat(PropGridLineCount, gridLineCount);
            _domeMaterial.SetFloat(PropGridThickness, gridThickness);
            _domeMaterial.SetInt(PropCull, (int)UnityEngine.Rendering.CullMode.Off);

            if (voxelWorld != null)
            {
                float extent = voxelWorld.WorldExtent;
                Vector3 center = voxelWorld.WorldOrigin + Vector3.one * extent * 0.5f;
                _domeMaterial.SetVector(PropDomeCenter, center);
                _domeMaterial.SetFloat(PropDomeExtent, extent);
                _domeMaterial.SetFloat(PropDomeMargin, normalizedMargin);
            }
        }

        private void UpdateVisibility()
        {
            if (_meshRenderer == null || dualCamera == null) return;

            bool shouldShow = dualCamera.CameraMode == VoxelCameraMode.GodView
                ? visibleInGodView
                : visibleInInsideView;

            if (_meshRenderer.enabled != shouldShow)
                _meshRenderer.enabled = shouldShow;
        }

        // =================================================================
        // Mesh
        // =================================================================

        /// <summary>
        /// Create a cube mesh matching VoxelRenderer's coordinate system.
        /// Vertices go from [-margin, 1+margin] so the glass sits just
        /// outside the voxel volume. With scale = worldExtent and
        /// position = WorldOrigin, this perfectly wraps the map.
        /// </summary>
        private static Mesh CreateDomeMesh(float margin)
        {
            var mesh = new Mesh();
            mesh.name = "GlassDomeCube";

            float lo = -margin;
            float hi = 1f + margin;

            Vector3[] verts =
            {
                // Front face (+Z)
                new Vector3(lo, lo, hi), new Vector3(hi, lo, hi),
                new Vector3(hi, hi, hi), new Vector3(lo, hi, hi),
                // Back face (-Z)
                new Vector3(hi, lo, lo), new Vector3(lo, lo, lo),
                new Vector3(lo, hi, lo), new Vector3(hi, hi, lo),
                // Top (+Y)
                new Vector3(lo, hi, hi), new Vector3(hi, hi, hi),
                new Vector3(hi, hi, lo), new Vector3(lo, hi, lo),
                // Bottom (-Y)
                new Vector3(lo, lo, lo), new Vector3(hi, lo, lo),
                new Vector3(hi, lo, hi), new Vector3(lo, lo, hi),
                // Right (+X)
                new Vector3(hi, lo, hi), new Vector3(hi, lo, lo),
                new Vector3(hi, hi, lo), new Vector3(hi, hi, hi),
                // Left (-X)
                new Vector3(lo, lo, lo), new Vector3(lo, lo, hi),
                new Vector3(lo, hi, hi), new Vector3(lo, hi, lo),
            };

            Vector3[] normals =
            {
                Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward,
                Vector3.back,    Vector3.back,    Vector3.back,    Vector3.back,
                Vector3.up,      Vector3.up,      Vector3.up,      Vector3.up,
                Vector3.down,    Vector3.down,    Vector3.down,    Vector3.down,
                Vector3.right,   Vector3.right,   Vector3.right,   Vector3.right,
                Vector3.left,    Vector3.left,    Vector3.left,    Vector3.left,
            };

            int[] tris =
            {
                 0, 2, 1,  0, 3, 2,
                 4, 6, 5,  4, 7, 6,
                 8,10, 9,  8,11,10,
                12,14,13, 12,15,14,
                16,18,17, 16,19,18,
                20,22,21, 20,23,22,
            };

            mesh.vertices = verts;
            mesh.normals = normals;
            mesh.triangles = tris;
            mesh.RecalculateBounds();

            return mesh;
        }

        // =================================================================
        // Gizmos
        // =================================================================

        private void OnDrawGizmosSelected()
        {
            VoxelWorld vw = voxelWorld;
            if (vw == null) vw = GetComponent<VoxelWorld>();
            if (vw == null) return;

            float extent = vw.WorldExtent;
            float m = normalizedMargin * extent;
            Vector3 center = vw.WorldOrigin + Vector3.one * extent * 0.5f;
            float totalSize = extent + m * 2f;

            Gizmos.color = new Color(0.4f, 0.7f, 1f, 0.4f);
            Gizmos.DrawWireCube(center, Vector3.one * totalSize);
        }
    }
}
