using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace VoxelEngine
{
    /// <summary>
    /// Draws auxiliary world objects (debris/props/chunk debug) via DrawMeshInstancedIndirect.
    /// Each group is rendered in one draw call and can be bound to a single material.
    /// </summary>
    public class VoxelIndirectInstanceRenderer : MonoBehaviour
    {
        [Serializable]
        public class InstanceGroup
        {
            public string name = "Group";
            public Mesh mesh;
            public Material material;
            public List<Transform> instances = new List<Transform>();
            public int subMeshIndex;
            public bool enabled = true;
            public bool autoBoundsFromInstances = true;
            public Vector3 manualBoundsCenter;
            public Vector3 manualBoundsSize = new Vector3(256f, 256f, 256f);
            [Range(0, 31)] public int layer;
        }

        private sealed class RuntimeGroup
        {
            public GraphicsBuffer matrixBuffer;
            public GraphicsBuffer argsBuffer;
            public MaterialPropertyBlock mpb;
            public readonly uint[] args = new uint[5];
            public int allocatedCount;
        }

        private readonly struct DrawKey : IEquatable<DrawKey>
        {
            public readonly Mesh mesh;
            public readonly Material material;
            public readonly int subMesh;
            public readonly int layer;

            public DrawKey(Mesh mesh, Material material, int subMesh, int layer)
            {
                this.mesh = mesh;
                this.material = material;
                this.subMesh = subMesh;
                this.layer = layer;
            }

            public bool Equals(DrawKey other)
            {
                return mesh == other.mesh && material == other.material && subMesh == other.subMesh && layer == other.layer;
            }

            public override bool Equals(object obj)
            {
                return obj is DrawKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = 17;
                    hash = hash * 31 + (mesh != null ? mesh.GetHashCode() : 0);
                    hash = hash * 31 + (material != null ? material.GetHashCode() : 0);
                    hash = hash * 31 + subMesh;
                    hash = hash * 31 + layer;
                    return hash;
                }
            }
        }

        private static readonly int PropInstanceMatrices = Shader.PropertyToID("_InstanceMatrices");

        [Header("Indirect Instancing")]
        [SerializeField] private bool drawInPlayMode = true;
        [SerializeField] private bool drawInEditMode = true;
        [SerializeField] private ShadowCastingMode shadowCastingMode = ShadowCastingMode.Off;
        [SerializeField] private bool receiveShadows = false;

        [Header("Auto Source Batching")]
        [SerializeField] private bool autoCollectSources = true;
        [SerializeField] private Transform sourceRoot;
        [SerializeField] private bool includeInactiveSources = false;
        [SerializeField] private bool autoBoundsFromSources = true;
        [SerializeField] private Vector3 fallbackAutoBoundsSize = new Vector3(256f, 256f, 256f);

        [Header("Manual Groups")]
        [SerializeField] private List<InstanceGroup> groups = new List<InstanceGroup>();

        private readonly List<Matrix4x4> _matrixScratch = new List<Matrix4x4>(1024);
        private RuntimeGroup[] _runtimeGroups;
        private readonly Dictionary<DrawKey, List<Matrix4x4>> _autoGroupedMatrices = new Dictionary<DrawKey, List<Matrix4x4>>();
        private readonly Dictionary<DrawKey, RuntimeGroup> _autoRuntimeGroups = new Dictionary<DrawKey, RuntimeGroup>();
        private readonly HashSet<DrawKey> _activeAutoKeys = new HashSet<DrawKey>();

        public List<InstanceGroup> Groups => groups;

        private void OnEnable()
        {
            EnsureRuntimeStorage();
        }

        private void OnDisable()
        {
            ReleaseRuntimeBuffers();
            ReleaseAutoRuntimeBuffers();
        }

        private void OnValidate()
        {
            EnsureRuntimeStorage();
        }

        private void LateUpdate()
        {
            if (Application.isPlaying && !drawInPlayMode)
                return;

            DrawAllGroups();
        }

        private void OnRenderObject()
        {
            if (Application.isPlaying || !drawInEditMode)
                return;

            DrawAllGroups();
        }

        private void EnsureRuntimeStorage()
        {
            if (groups == null)
                groups = new List<InstanceGroup>();

            if (_runtimeGroups != null && _runtimeGroups.Length == groups.Count)
                return;

            ReleaseRuntimeBuffers();
            _runtimeGroups = new RuntimeGroup[groups.Count];
            for (int i = 0; i < _runtimeGroups.Length; i++)
                _runtimeGroups[i] = new RuntimeGroup { mpb = new MaterialPropertyBlock() };
        }

        private void DrawAllGroups()
        {
            if (groups != null && groups.Count > 0)
            {
                EnsureRuntimeStorage();

                for (int i = 0; i < groups.Count; i++)
                    DrawGroup(i, groups[i], _runtimeGroups[i]);
            }

            if (autoCollectSources)
                DrawAutoCollectedSources();
        }

        private void DrawGroup(int index, InstanceGroup group, RuntimeGroup runtime)
        {
            if (group == null || !group.enabled)
                return;

            if (group.mesh == null || group.material == null)
                return;

            int subMesh = Mathf.Clamp(group.subMeshIndex, 0, Mathf.Max(0, group.mesh.subMeshCount - 1));

            _matrixScratch.Clear();
            CollectInstanceMatrices(group.instances, _matrixScratch);

            int instanceCount = _matrixScratch.Count;
            if (instanceCount == 0)
                return;

            EnsureCapacity(runtime, instanceCount);

            runtime.matrixBuffer.SetData(_matrixScratch);

            runtime.args[0] = group.mesh.GetIndexCount(subMesh);
            runtime.args[1] = (uint)instanceCount;
            runtime.args[2] = group.mesh.GetIndexStart(subMesh);
            runtime.args[3] = group.mesh.GetBaseVertex(subMesh);
            runtime.args[4] = 0;
            runtime.argsBuffer.SetData(runtime.args);

            runtime.mpb.Clear();
            runtime.mpb.SetBuffer(PropInstanceMatrices, runtime.matrixBuffer);

            if (!group.material.enableInstancing)
                group.material.enableInstancing = true;

            Bounds drawBounds = BuildBounds(group, group.mesh, _matrixScratch);

            Graphics.DrawMeshInstancedIndirect(
                group.mesh,
                subMesh,
                group.material,
                drawBounds,
                runtime.argsBuffer,
                0,
                runtime.mpb,
                shadowCastingMode,
                receiveShadows,
                group.layer
            );
        }

        private static void CollectInstanceMatrices(List<Transform> instances, List<Matrix4x4> outMatrices)
        {
            if (instances == null)
                return;

            int count = instances.Count;
            for (int i = 0; i < count; i++)
            {
                Transform t = instances[i];
                if (t == null)
                    continue;

                outMatrices.Add(t.localToWorldMatrix);
            }
        }

        private static Bounds BuildBounds(InstanceGroup group, Mesh mesh, List<Matrix4x4> matrices)
        {
            if (!group.autoBoundsFromInstances)
                return new Bounds(group.manualBoundsCenter, Vector3.Max(group.manualBoundsSize, Vector3.one * 0.1f));

            if (matrices == null || matrices.Count == 0)
                return new Bounds(group.manualBoundsCenter, Vector3.Max(group.manualBoundsSize, Vector3.one * 0.1f));

            Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
            Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

            Vector3 meshExtents = mesh.bounds.extents;
            float extMagnitude = meshExtents.magnitude;

            for (int i = 0; i < matrices.Count; i++)
            {
                Matrix4x4 m = matrices[i];
                Vector3 p = m.GetColumn(3);

                float sx = new Vector3(m.m00, m.m10, m.m20).magnitude;
                float sy = new Vector3(m.m01, m.m11, m.m21).magnitude;
                float sz = new Vector3(m.m02, m.m12, m.m22).magnitude;
                float scale = Mathf.Max(sx, Mathf.Max(sy, sz));

                Vector3 e = Vector3.one * Mathf.Max(0.01f, extMagnitude * scale);
                min = Vector3.Min(min, p - e);
                max = Vector3.Max(max, p + e);
            }

            Vector3 size = Vector3.Max(max - min, Vector3.one * 0.1f);
            Vector3 center = (min + max) * 0.5f;
            return new Bounds(center, size);
        }

        private void DrawAutoCollectedSources()
        {
            foreach (var kvp in _autoGroupedMatrices)
                kvp.Value.Clear();

            _activeAutoKeys.Clear();

            Transform root = sourceRoot != null ? sourceRoot : transform;
            VoxelIndirectInstanceSource[] sources = root.GetComponentsInChildren<VoxelIndirectInstanceSource>(includeInactiveSources);
            for (int i = 0; i < sources.Length; i++)
            {
                VoxelIndirectInstanceSource source = sources[i];
                if (source == null || !source.IsRenderable(includeInactiveSources))
                    continue;

                if (!source.TryGetDrawData(out Mesh mesh, out Material material, out int subMeshIndex, out int layer))
                    continue;

                int subMesh = Mathf.Clamp(subMeshIndex, 0, Mathf.Max(0, mesh.subMeshCount - 1));
                int instanceLayer = Mathf.Clamp(layer, 0, 31);

                DrawKey key = new DrawKey(mesh, material, subMesh, instanceLayer);
                if (!_autoGroupedMatrices.TryGetValue(key, out List<Matrix4x4> list))
                {
                    list = new List<Matrix4x4>(64);
                    _autoGroupedMatrices.Add(key, list);
                }

                list.Add(source.transform.localToWorldMatrix);
                _activeAutoKeys.Add(key);
            }

            foreach (var kvp in _autoGroupedMatrices)
            {
                DrawKey key = kvp.Key;
                List<Matrix4x4> matrices = kvp.Value;
                int instanceCount = matrices.Count;
                if (instanceCount == 0)
                    continue;

                if (!_autoRuntimeGroups.TryGetValue(key, out RuntimeGroup runtime) || runtime == null)
                {
                    runtime = new RuntimeGroup { mpb = new MaterialPropertyBlock() };
                    _autoRuntimeGroups[key] = runtime;
                }

                EnsureCapacity(runtime, instanceCount);
                runtime.matrixBuffer.SetData(matrices);

                runtime.args[0] = key.mesh.GetIndexCount(key.subMesh);
                runtime.args[1] = (uint)instanceCount;
                runtime.args[2] = key.mesh.GetIndexStart(key.subMesh);
                runtime.args[3] = key.mesh.GetBaseVertex(key.subMesh);
                runtime.args[4] = 0;
                runtime.argsBuffer.SetData(runtime.args);

                runtime.mpb.Clear();
                runtime.mpb.SetBuffer(PropInstanceMatrices, runtime.matrixBuffer);

                if (!key.material.enableInstancing)
                    key.material.enableInstancing = true;

                Bounds bounds = autoBoundsFromSources
                    ? BuildAutoBounds(key.mesh, matrices)
                    : new Bounds(root.position, Vector3.Max(fallbackAutoBoundsSize, Vector3.one * 0.1f));

                Graphics.DrawMeshInstancedIndirect(
                    key.mesh,
                    key.subMesh,
                    key.material,
                    bounds,
                    runtime.argsBuffer,
                    0,
                    runtime.mpb,
                    shadowCastingMode,
                    receiveShadows,
                    key.layer
                );
            }

            ReleaseStaleAutoRuntimeBuffers();
        }

        private static Bounds BuildAutoBounds(Mesh mesh, List<Matrix4x4> matrices)
        {
            Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
            Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

            Vector3 meshExtents = mesh.bounds.extents;
            float extMagnitude = meshExtents.magnitude;

            for (int i = 0; i < matrices.Count; i++)
            {
                Matrix4x4 m = matrices[i];
                Vector3 p = m.GetColumn(3);

                float sx = new Vector3(m.m00, m.m10, m.m20).magnitude;
                float sy = new Vector3(m.m01, m.m11, m.m21).magnitude;
                float sz = new Vector3(m.m02, m.m12, m.m22).magnitude;
                float scale = Mathf.Max(sx, Mathf.Max(sy, sz));

                Vector3 e = Vector3.one * Mathf.Max(0.01f, extMagnitude * scale);
                min = Vector3.Min(min, p - e);
                max = Vector3.Max(max, p + e);
            }

            Vector3 size = Vector3.Max(max - min, Vector3.one * 0.1f);
            Vector3 center = (min + max) * 0.5f;
            return new Bounds(center, size);
        }

        private static void EnsureCapacity(RuntimeGroup runtime, int requiredInstances)
        {
            int needed = Mathf.NextPowerOfTwo(Mathf.Max(1, requiredInstances));
            if (runtime.allocatedCount >= needed && runtime.matrixBuffer != null && runtime.argsBuffer != null)
                return;

            runtime.matrixBuffer?.Release();
            runtime.argsBuffer?.Release();

            runtime.matrixBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, needed, sizeof(float) * 16);
            runtime.argsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, sizeof(uint) * 5);
            runtime.allocatedCount = needed;
        }

        private void ReleaseRuntimeBuffers()
        {
            if (_runtimeGroups == null)
                return;

            for (int i = 0; i < _runtimeGroups.Length; i++)
            {
                RuntimeGroup runtime = _runtimeGroups[i];
                if (runtime == null)
                    continue;

                runtime.matrixBuffer?.Release();
                runtime.argsBuffer?.Release();
                runtime.matrixBuffer = null;
                runtime.argsBuffer = null;
                runtime.allocatedCount = 0;
            }

            _runtimeGroups = null;
        }

        private void ReleaseStaleAutoRuntimeBuffers()
        {
            if (_autoRuntimeGroups.Count == 0)
                return;

            List<DrawKey> staleKeys = null;
            foreach (var kvp in _autoRuntimeGroups)
            {
                if (_activeAutoKeys.Contains(kvp.Key))
                    continue;

                staleKeys ??= new List<DrawKey>();
                staleKeys.Add(kvp.Key);
            }

            if (staleKeys == null)
                return;

            for (int i = 0; i < staleKeys.Count; i++)
            {
                DrawKey key = staleKeys[i];
                RuntimeGroup runtime = _autoRuntimeGroups[key];
                runtime.matrixBuffer?.Release();
                runtime.argsBuffer?.Release();
                _autoRuntimeGroups.Remove(key);
            }
        }

        private void ReleaseAutoRuntimeBuffers()
        {
            foreach (var kvp in _autoRuntimeGroups)
            {
                RuntimeGroup runtime = kvp.Value;
                if (runtime == null)
                    continue;

                runtime.matrixBuffer?.Release();
                runtime.argsBuffer?.Release();
            }

            _autoRuntimeGroups.Clear();
            _autoGroupedMatrices.Clear();
            _activeAutoKeys.Clear();
        }

        public void SetGroupInstances(int groupIndex, List<Transform> newInstances)
        {
            if (groupIndex < 0 || groupIndex >= groups.Count)
                return;

            groups[groupIndex].instances = newInstances ?? new List<Transform>();
        }

        public int AddGroup(string groupName, Mesh mesh, Material material)
        {
            var group = new InstanceGroup
            {
                name = string.IsNullOrWhiteSpace(groupName) ? "Group" : groupName,
                mesh = mesh,
                material = material
            };
            groups.Add(group);
            EnsureRuntimeStorage();
            return groups.Count - 1;
        }
    }
}
