using UnityEngine;
using System.Collections.Generic;

namespace VoxelEngine
{
    /// <summary>
    /// Generates collision meshes from voxel data around the player using Greedy Meshing.
    /// Outputs to a MeshCollider for physics interaction with Unity's physics system.
    /// </summary>
    [RequireComponent(typeof(MeshCollider))]
    public class VoxelCollision : MonoBehaviour
    {
        [Header("Configuration")]
        [SerializeField] private VoxelWorld voxelWorld;
        [SerializeField] private Transform trackTarget; // Usually the player
        [SerializeField] private bool enableCollisionMeshing = false;
        [SerializeField] private int collisionRadius = 16; // Voxels around target
        [SerializeField] private float updateInterval = 0.25f; // Seconds between updates
        [SerializeField] [Min(1)] private int movementThreshold = 2;
        [SerializeField] private bool debugDraw = false;

        private MeshCollider _meshCollider;
        private Mesh _collisionMesh;
        private float _lastUpdateTime;
        private Vector3Int _lastUpdateCenter;
        private uint[] _readbackData;
        private bool _needsUpdate;
        private bool _hasAssignedMesh;

        // Greedy mesh data
        private List<Vector3> _vertices = new List<Vector3>();
        private List<int> _triangles = new List<int>();

        private void Awake()
        {
            _meshCollider = GetComponent<MeshCollider>();
            _collisionMesh = new Mesh();
            _collisionMesh.name = "VoxelCollisionMesh";
            _meshCollider.sharedMesh = _collisionMesh;
        }

        private void Start()
        {
            if (voxelWorld == null)
                voxelWorld = FindFirstObjectByType<VoxelWorld>();
        }

        private void Update()
        {
            if (!enableCollisionMeshing) return;
            if (voxelWorld == null || trackTarget == null) return;
            if (!Application.isPlaying) return;

            if (Time.time - _lastUpdateTime < updateInterval) return;

            Vector3Int currentCenter = voxelWorld.WorldToVoxel(trackTarget.position);

            // Only update if player moved significantly
            if ((currentCenter - _lastUpdateCenter).sqrMagnitude < movementThreshold * movementThreshold && !_needsUpdate) return;

            _lastUpdateTime = Time.time;
            _lastUpdateCenter = currentCenter;
            _needsUpdate = false;

            GenerateCollisionMesh(currentCenter);
        }

        public void ForceUpdate()
        {
            _needsUpdate = true;
        }

        private void GenerateCollisionMesh(Vector3Int center)
        {
            if (voxelWorld.ReadBuffer == null) return;

            // Shared throttled readback from VoxelWorld
            _readbackData = voxelWorld.GetCpuVoxelData();
            if (_readbackData == null) return;

            // Build collision mesh using Greedy Meshing
            _vertices.Clear();
            _triangles.Clear();

            int ws = voxelWorld.WorldSize;
            int minX = Mathf.Max(0, center.x - collisionRadius);
            int maxX = Mathf.Min(ws, center.x + collisionRadius);
            int minY = Mathf.Max(0, center.y - collisionRadius);
            int maxY = Mathf.Min(ws, center.y + collisionRadius);
            int minZ = Mathf.Max(0, center.z - collisionRadius);
            int maxZ = Mathf.Min(ws, center.z + collisionRadius);

            // For each of the 6 face directions, run greedy meshing
            GreedyMeshAxis(minX, maxX, minY, maxY, minZ, maxZ, ws, 0); // +X
            GreedyMeshAxis(minX, maxX, minY, maxY, minZ, maxZ, ws, 1); // -X
            GreedyMeshAxis(minX, maxX, minY, maxY, minZ, maxZ, ws, 2); // +Y
            GreedyMeshAxis(minX, maxX, minY, maxY, minZ, maxZ, ws, 3); // -Y
            GreedyMeshAxis(minX, maxX, minY, maxY, minZ, maxZ, ws, 4); // +Z
            GreedyMeshAxis(minX, maxX, minY, maxY, minZ, maxZ, ws, 5); // -Z

            // Update mesh
            if (_vertices.Count == 0 || _triangles.Count == 0)
            {
                if (_hasAssignedMesh)
                {
                    _meshCollider.sharedMesh = null;
                    _hasAssignedMesh = false;
                }
                return;
            }

            _collisionMesh.Clear();
            _collisionMesh.SetVertices(_vertices);
            _collisionMesh.SetTriangles(_triangles, 0);
            _collisionMesh.RecalculateBounds();

            _meshCollider.sharedMesh = null;
            _meshCollider.sharedMesh = _collisionMesh;
            _hasAssignedMesh = true;
        }

        private void GreedyMeshAxis(int minX, int maxX, int minY, int maxY, int minZ, int maxZ, int ws, int face)
        {
            // Face directions:
            // 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
            // For simplicity, generate individual quads for exposed faces
            // A full greedy meshing would merge adjacent faces, but this is sufficient for collision

            float scale = voxelWorld.VoxelScale;
            Vector3 origin = voxelWorld.WorldOrigin;

            for (int z = minZ; z < maxZ; z++)
            for (int y = minY; y < maxY; y++)
            for (int x = minX; x < maxX; x++)
            {
                int idx = VoxelData.Flatten3D(x, y, z, ws);
                uint mat = VoxelData.GetMaterialId(_readbackData[idx]);

                if (!VoxelData.IsSolid(mat)) continue;

                // Check if this face is exposed (neighbor is air or out of bounds)
                int nx = x, ny = y, nz = z;
                switch (face)
                {
                    case 0: nx = x + 1; break;
                    case 1: nx = x - 1; break;
                    case 2: ny = y + 1; break;
                    case 3: ny = y - 1; break;
                    case 4: nz = z + 1; break;
                    case 5: nz = z - 1; break;
                }

                bool exposed = false;
                if (nx < 0 || nx >= ws || ny < 0 || ny >= ws || nz < 0 || nz >= ws)
                {
                    exposed = true;
                }
                else
                {
                    uint neighborMat = VoxelData.GetMaterialId(_readbackData[VoxelData.Flatten3D(nx, ny, nz, ws)]);
                    exposed = !VoxelData.IsSolid(neighborMat);
                }

                if (!exposed) continue;

                // Add quad for this face
                AddFaceQuad(x, y, z, face, scale, origin);
            }
        }

        private void AddFaceQuad(int x, int y, int z, int face, float scale, Vector3 worldOrigin)
        {
            int baseVertex = _vertices.Count;
            Vector3 p = new Vector3(x, y, z) * scale + worldOrigin;
            float s = scale;

            switch (face)
            {
                case 0: // +X
                    _vertices.Add(p + new Vector3(s, 0, 0));
                    _vertices.Add(p + new Vector3(s, s, 0));
                    _vertices.Add(p + new Vector3(s, s, s));
                    _vertices.Add(p + new Vector3(s, 0, s));
                    break;
                case 1: // -X
                    _vertices.Add(p + new Vector3(0, 0, s));
                    _vertices.Add(p + new Vector3(0, s, s));
                    _vertices.Add(p + new Vector3(0, s, 0));
                    _vertices.Add(p + new Vector3(0, 0, 0));
                    break;
                case 2: // +Y
                    _vertices.Add(p + new Vector3(0, s, 0));
                    _vertices.Add(p + new Vector3(0, s, s));
                    _vertices.Add(p + new Vector3(s, s, s));
                    _vertices.Add(p + new Vector3(s, s, 0));
                    break;
                case 3: // -Y
                    _vertices.Add(p + new Vector3(0, 0, s));
                    _vertices.Add(p + new Vector3(0, 0, 0));
                    _vertices.Add(p + new Vector3(s, 0, 0));
                    _vertices.Add(p + new Vector3(s, 0, s));
                    break;
                case 4: // +Z
                    _vertices.Add(p + new Vector3(s, 0, s));
                    _vertices.Add(p + new Vector3(s, s, s));
                    _vertices.Add(p + new Vector3(0, s, s));
                    _vertices.Add(p + new Vector3(0, 0, s));
                    break;
                case 5: // -Z
                    _vertices.Add(p + new Vector3(0, 0, 0));
                    _vertices.Add(p + new Vector3(0, s, 0));
                    _vertices.Add(p + new Vector3(s, s, 0));
                    _vertices.Add(p + new Vector3(s, 0, 0));
                    break;
            }

            _triangles.Add(baseVertex);
            _triangles.Add(baseVertex + 1);
            _triangles.Add(baseVertex + 2);
            _triangles.Add(baseVertex);
            _triangles.Add(baseVertex + 2);
            _triangles.Add(baseVertex + 3);
        }

        private void OnDrawGizmosSelected()
        {
            if (!debugDraw || voxelWorld == null || trackTarget == null) return;

            Vector3Int center = voxelWorld.WorldToVoxel(trackTarget.position);
            float s = voxelWorld.VoxelScale;
            Vector3 o = voxelWorld.WorldOrigin;

            Vector3 min = new Vector3(
                (center.x - collisionRadius) * s + o.x,
                (center.y - collisionRadius) * s + o.y,
                (center.z - collisionRadius) * s + o.z
            );
            Vector3 max = new Vector3(
                (center.x + collisionRadius) * s + o.x,
                (center.y + collisionRadius) * s + o.y,
                (center.z + collisionRadius) * s + o.z
            );

            Gizmos.color = new Color(1, 0.5f, 0, 0.3f);
            Gizmos.DrawWireCube((min + max) / 2, max - min);
        }
    }
}
