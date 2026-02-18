using UnityEngine;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace VoxelEngine
{
    /// <summary>
    /// Generates collision meshes from voxel data around the player using Greedy Meshing.
    /// Outputs to a MeshCollider for physics interaction with Unity's physics system.
    /// </summary>
    [RequireComponent(typeof(MeshCollider))]
    public class VoxelCollision : MonoBehaviour
    {
        private const uint MatAir = 0u;
        private const uint MatWater = 5u;
        private const uint MatLava = 6u;
        private const uint MatSteam = 14u;

        private struct FaceRecord
        {
            public int x;
            public int y;
            public int z;
            public byte face;
        }

        [BurstCompile]
        private struct CollectExposedFacesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> paddedVoxelData;
            public NativeStream.Writer writer;

            public int worldMinX;
            public int worldMinY;
            public int worldMinZ;

            public int sizeX;
            public int sizeY;
            public int sizeZ;

            public int paddedSizeX;
            public int paddedSizeY;

            public void Execute(int index)
            {
                writer.BeginForEachIndex(index);

                int xy = sizeX * sizeY;
                int localZ = index / xy;
                int rem = index - localZ * xy;
                int localY = rem / sizeX;
                int localX = rem - localY * sizeX;

                int px = localX + 1;
                int py = localY + 1;
                int pz = localZ + 1;
                int baseIndex = Flatten3D(px, py, pz, paddedSizeX, paddedSizeY);

                uint mat = paddedVoxelData[baseIndex] & 0xFFu;
                if (!IsSolid(mat))
                {
                    writer.EndForEachIndex();
                    return;
                }

                int wx = worldMinX + localX;
                int wy = worldMinY + localY;
                int wz = worldMinZ + localZ;

                if (IsAirAt(baseIndex + 1))
                    WriteFace(wx, wy, wz, 0);
                if (IsAirAt(baseIndex - 1))
                    WriteFace(wx, wy, wz, 1);
                if (IsAirAt(baseIndex + paddedSizeX))
                    WriteFace(wx, wy, wz, 2);
                if (IsAirAt(baseIndex - paddedSizeX))
                    WriteFace(wx, wy, wz, 3);

                int zStride = paddedSizeX * paddedSizeY;
                if (IsAirAt(baseIndex + zStride))
                    WriteFace(wx, wy, wz, 4);
                if (IsAirAt(baseIndex - zStride))
                    WriteFace(wx, wy, wz, 5);

                writer.EndForEachIndex();
            }

            private void WriteFace(int x, int y, int z, byte face)
            {
                FaceRecord rec;
                rec.x = x;
                rec.y = y;
                rec.z = z;
                rec.face = face;
                writer.Write(rec);
            }

            private bool IsAirAt(int idx)
            {
                uint neighborMat = paddedVoxelData[idx] & 0xFFu;
                return !IsSolid(neighborMat);
            }

            private static bool IsSolid(uint materialId)
            {
                return materialId != MatAir && materialId != MatWater &&
                       materialId != MatLava && materialId != MatSteam;
            }

            private static int Flatten3D(int x, int y, int z, int sx, int sy)
            {
                return x + y * sx + z * sx * sy;
            }
        }

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
        private int _lastTopologyHash;
        private int _lastFaceCount = -1;
        private bool _hasTopologyCache;

        // Greedy mesh data
        private List<Vector3> _vertices = new List<Vector3>();
        private List<int> _triangles = new List<int>();

        private void Awake()
        {
            _meshCollider = GetComponent<MeshCollider>();
            _collisionMesh = new Mesh();
            _collisionMesh.name = "VoxelCollisionMesh";
            _collisionMesh.MarkDynamic();
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

            // ---- Fast path: read from CPU NativeArray directly (no managed copy) ----
            bool useCpuNative = voxelWorld.TryGetCpuVoxelNativeData(out NativeArray<uint> cpuVoxelNative);

            if (!useCpuNative)
            {
                // Fallback: GPU readback via managed array
                _readbackData = voxelWorld.GetCpuVoxelData();
                if (_readbackData == null) return;
            }

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

            int sizeX = Mathf.Max(0, maxX - minX);
            int sizeY = Mathf.Max(0, maxY - minY);
            int sizeZ = Mathf.Max(0, maxZ - minZ);
            if (sizeX == 0 || sizeY == 0 || sizeZ == 0)
            {
                if (_hasAssignedMesh)
                {
                    _meshCollider.sharedMesh = null;
                    _hasAssignedMesh = false;
                }
                return;
            }

            int paddedSizeX = sizeX + 2;
            int paddedSizeY = sizeY + 2;
            int paddedSizeZ = sizeZ + 2;
            int paddedTotal = paddedSizeX * paddedSizeY * paddedSizeZ;

            var paddedData = new NativeArray<uint>(paddedTotal, Allocator.TempJob, NativeArrayOptions.ClearMemory);

            try
            {
                if (useCpuNative)
                {
                    // Parallel fill via Burst job — reads NativeArray directly (zero managed copies)
                    var fillJob = new FillPaddedVoxelDataJob
                    {
                        worldData = cpuVoxelNative,
                        paddedData = paddedData,
                        paddedSizeX = paddedSizeX,
                        paddedSizeY = paddedSizeY,
                        minX = minX,
                        minY = minY,
                        minZ = minZ,
                        worldSize = ws
                    };
                    fillJob.Schedule(paddedTotal, 256).Complete();
                }
                else
                {
                    // Serial fill from managed array (GPU readback path)
                    FillPaddedVoxelData(paddedData, paddedSizeX, paddedSizeY, paddedSizeZ,
                        minX, minY, minZ, ws);
                }

                int voxelCount = sizeX * sizeY * sizeZ;
                var stream = new NativeStream(voxelCount, Allocator.TempJob);

                try
                {
                    var job = new CollectExposedFacesJob
                    {
                        paddedVoxelData = paddedData,
                        writer = stream.AsWriter(),
                        worldMinX = minX,
                        worldMinY = minY,
                        worldMinZ = minZ,
                        sizeX = sizeX,
                        sizeY = sizeY,
                        sizeZ = sizeZ,
                        paddedSizeX = paddedSizeX,
                        paddedSizeY = paddedSizeY
                    };

                    JobHandle handle = job.Schedule(voxelCount, 128);
                    handle.Complete();

                    var reader = stream.AsReader();
                    float scale = voxelWorld.VoxelScale;
                    Vector3 origin = voxelWorld.WorldOrigin;

                    uint topologyHash = 2166136261u;
                    int faceCount = 0;

                    for (int i = 0; i < voxelCount; i++)
                    {
                        reader.BeginForEachIndex(i);
                        while (reader.RemainingItemCount > 0)
                        {
                            FaceRecord face = reader.Read<FaceRecord>();
                            faceCount++;
                            topologyHash = Fnv1a(topologyHash, face.x);
                            topologyHash = Fnv1a(topologyHash, face.y);
                            topologyHash = Fnv1a(topologyHash, face.z);
                            topologyHash = Fnv1a(topologyHash, face.face);
                            AddFaceQuad(face.x, face.y, face.z, face.face, scale, origin);
                        }
                        reader.EndForEachIndex();
                    }

                    int currentHash = unchecked((int)topologyHash);
                    if (_hasTopologyCache && _lastFaceCount == faceCount && _lastTopologyHash == currentHash)
                        return;

                    _lastFaceCount = faceCount;
                    _lastTopologyHash = currentHash;
                    _hasTopologyCache = true;
                }
                finally
                {
                    stream.Dispose();
                }
            }
            finally
            {
                paddedData.Dispose();
            }

            // Update mesh
            if (_vertices.Count == 0 || _triangles.Count == 0)
            {
                if (_hasAssignedMesh)
                {
                    _meshCollider.sharedMesh = null;
                    _hasAssignedMesh = false;
                }
                _hasTopologyCache = false;
                _lastFaceCount = -1;
                return;
            }

            _collisionMesh.Clear();
            _collisionMesh.SetVertices(_vertices);
            _collisionMesh.SetTriangles(_triangles, 0);

            float s = voxelWorld.VoxelScale;
            Vector3 o = voxelWorld.WorldOrigin;
            Vector3 bMin = new Vector3(minX * s + o.x, minY * s + o.y, minZ * s + o.z);
            Vector3 bMax = new Vector3(maxX * s + o.x, maxY * s + o.y, maxZ * s + o.z);
            _collisionMesh.bounds = new Bounds((bMin + bMax) * 0.5f, bMax - bMin);

            _meshCollider.sharedMesh = null;
            _meshCollider.sharedMesh = _collisionMesh;
            _hasAssignedMesh = true;
        }

        private static uint Fnv1a(uint hash, int value)
        {
            unchecked
            {
                hash ^= (uint)value;
                hash *= 16777619u;
                return hash;
            }
        }

        private void FillPaddedVoxelData(
            NativeArray<uint> paddedData,
            int paddedSizeX,
            int paddedSizeY,
            int paddedSizeZ,
            int minX,
            int minY,
            int minZ,
            int worldSize)
        {
            int zStride = paddedSizeX * paddedSizeY;
            for (int pz = 0; pz < paddedSizeZ; pz++)
            {
                int wz = minZ + pz - 1;
                for (int py = 0; py < paddedSizeY; py++)
                {
                    int wy = minY + py - 1;
                    for (int px = 0; px < paddedSizeX; px++)
                    {
                        int wx = minX + px - 1;
                        int pIdx = px + py * paddedSizeX + pz * zStride;

                        if (wx < 0 || wx >= worldSize || wy < 0 || wy >= worldSize || wz < 0 || wz >= worldSize)
                        {
                            paddedData[pIdx] = 0;
                            continue;
                        }

                        int worldIdx = VoxelData.Flatten3D(wx, wy, wz, worldSize);
                        paddedData[pIdx] = _readbackData[worldIdx];
                    }
                }
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
